from abc import abstractmethod
from tissue_classes import Tissue
import cv2
import numpy as np
import pywt
import json
import SimpleITK as sitk
from sklearn.decomposition import PCA

class Register:
    @abstractmethod
    def register(self, tissue1: Tissue, tissue2: Tissue):
        pass

    @abstractmethod
    def find_transform(self, tissue1: Tissue, tissue2: Tissue):
        pass

class TissueAffineRegister(Register):
    def _power(self, l, r1, r2):
        r = (r1 + r2) / 2
        return (l + 1) * (r ** 1.5) + 1e-6

    def _linear(self, l, r1, r2):
        r = (r1 + r2) / 2
        return (l + 1) * r / 2 + 1e-6

    def _he_bior_pyramid(self, tissue, levels):
        def normalize(cA, mask, ref_max, ref_min):
            cA = (cA - cA.min()) / (cA.max() - cA.min() + 1e-12)
            cA = np.clip(cA - np.median(cA[~mask]), 0, 1)
            cA = (cA * (ref_max - ref_min) + ref_min).astype(np.uint8)
            return cA
        
        bior = [tissue.he]
        masks = [tissue.mask]

        hem_ref_max, hem_ref_min = tissue.hem.max(), tissue.hem.min()
        eos_ref_max, eos_ref_min = tissue.eos.max(), tissue.eos.min()

        for i in range(levels-1):
            cur_tissue = bior[i]
            cur_mask = masks[i]

            cA_hem, _ = pywt.dwt2(cur_tissue[...,0], 'bior6.8')
            cA_eos, _ = pywt.dwt2(cur_tissue[...,1], 'bior6.8')
            new_mask = cv2.resize(
                cur_mask.astype(np.uint8),
                (cA_hem.shape[1], cA_hem.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            cA_hem = normalize(cA_hem, new_mask, hem_ref_max, hem_ref_min)
            cA_eos = normalize(cA_eos, new_mask, eos_ref_max, eos_ref_min)

            bior.append(np.stack([cA_hem, cA_eos], axis=-1))
            masks.append(new_mask)

        return bior, masks

    def _scale_kp(kp: list[cv2.KeyPoint], scale: float) -> list[cv2.KeyPoint]:
        scaled_kp = []
        for k in kp:
            new_k = cv2.KeyPoint(k.pt[0] * scale, k.pt[1] * scale, k.size * scale, k.angle, k.response, k.octave, k.class_id)
            scaled_kp.append(new_k)
        return scaled_kp

    def _merge_matches(self, match1: list[cv2.DMatch], match2: list[cv2.DMatch], q_offset: int, t_offset: int):
        assert q_offset >= 0 and t_offset >= 0

        adjusted_match2 = [
            cv2.DMatch(m.queryIdx + q_offset, m.trainIdx + t_offset, m.distance)
            for m in match2
        ]

        return match1 + adjusted_match2
    
    def _find_scaled(
            self,
            feature: cv2.Feature2D,
            o_img1: np.ndarray, img1: np.ndarray, mask1: np.ndarray, 
            o_img2: np.ndarray, img2: np.ndarray, mask2: np.ndarray,
            return_params=False
        ):
        
        params = None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))

        mask1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=1).astype(bool)
        mask2 = cv2.dilate(mask2.astype(np.uint8), kernel, iterations=1).astype(bool)

        kp1, des1 = feature.detectAndCompute(img1, mask1.astype(np.uint8))
        kp2, des2 = feature.detectAndCompute(img2, mask2.astype(np.uint8))
        
        img1_scale = o_img1.shape[0] / img1.shape[0]
        img2_scale = o_img2.shape[0] / img2.shape[0]
        
        kp1 = self._scale_kp(kp1, img1_scale)
        kp2 = self._scale_kp(kp2, img2_scale)
        
        if return_params:
            params = {
                'kp1': kp1,
                'kp2': kp2,
                'des1': des1,
                'des2': des2,
            }

        return kp1, des1, kp2, des2, params
    
    def _lowes_test_ratio(self, matches: list[cv2.DMatch], ratio: float = 0.75) -> list[cv2.DMatch]:
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
        return good_matches

    def _filter_mask(self, kp: list[cv2.KeyPoint], mask: np.ndarray) -> list[cv2.KeyPoint]:
        filtered_kp = []
        for k in kp:
            if mask[int(k.pt[1]), int(k.pt[0])]:
                filtered_kp.append(k)
        return filtered_kp
    
    def _mutual_filter(self, matcher, des1, des2, good_matches, lowes_ratio=0.75):
        matches_rev = matcher.knnMatch(des2, des1, k=2)
        good_matches_rev = self._lowes_test_ratio(matches_rev, ratio=lowes_ratio)
        rev_pairs = set((m_rev.trainIdx, m_rev.queryIdx) for m_rev in good_matches_rev)
        filtered_matches = [m for m in good_matches if (m.queryIdx, m.trainIdx) in rev_pairs]
        return filtered_matches
    
    def _sample_matches(
            self,
            matches: list[cv2.DMatch], 
            kp1: list[cv2.KeyPoint], kp2: list[cv2.KeyPoint], 
            des1: list, des2: list,
            weights: np.ndarray[np.float32],
            sample_prc: float = 0.2,
            seed: int = 42,
            verbose: int = 0,
            return_params=False
        ):


        params = None
        if return_params:
            params = {}

        assert len(weights) == len(matches)
        
        if verbose >= 1:
            print(f'Total matches: {len(matches)}')
            print(f'Sampling {sample_prc*100:.1f}% of matches')
        
        np.random.seed(seed)
        w_min = weights.min()
        w_max = weights.max()
        if w_max == w_min:
            w_norm = np.ones_like(weights) / len(weights)
        else:
            w_norm = (weights - w_min) / (w_max - w_min)
            w_norm = w_norm / w_norm.sum()
        
        indexes = np.arange(len(weights))
        choice = np.random.choice(indexes, size=int(len(weights) * sample_prc), p=w_norm, replace=False)
        
        if verbose >= 1:
            print(f'Selected {len(choice)} matches after weighted sampling')
        
        ret_kp1 = []
        ret_kp2 = []
        ret_des1 = []
        ret_des2 = []
        ret_matches = []
        for i, c in enumerate(choice):
            q_idx = matches[c].queryIdx
            t_idx = matches[c].trainIdx
            
            ret_matches.append(matches[c])
            ret_kp1.append(kp1[q_idx])
            ret_kp2.append(kp2[t_idx])
            ret_des1.append(des1[q_idx])
            ret_des2.append(des2[t_idx])
            ret_matches[i].queryIdx = i
            ret_matches[i].trainIdx = i

        if verbose >= 1:
            print(f'Returned {len(ret_matches)} sampled matches')
        
        if return_params:
            params['sampled'] = ret_matches
            not_selected = list(set(np.arange(len(weights))) - set(choice))

            ret_matches_not = []
            ret_kp1_not = []
            ret_kp2_not = []
            ret_des1_not = []
            ret_des2_not = []

            for c in not_selected:
                q_idx = matches[c].queryIdx
                t_idx = matches[c].trainIdx

                ret_matches_not.append(matches[c])
                ret_kp1_not.append(kp1[q_idx])
                ret_kp2_not.append(kp2[t_idx])
                ret_des1_not.append(des1[q_idx])
                ret_des2_not.append(des2[t_idx])

            params['not_sampled'] = ret_matches_not

        return ret_matches, (ret_kp1, ret_des1), (ret_kp2, ret_des2), params

    def _filter_keypoints_and_matches(
        self,
        matcher, des1, des2, 
        lowes_ratio=0.75, mutual=False,
        verbose=0,
        return_params=False
    ):
        
        params = None
        if return_params:
            params = {
                'total': 0,
                'filtered': 0
            }

        if des1 is None or des2 is None:
            return [], params

        matches = matcher.knnMatch(des1, des2, k=2)
        
        if return_params:
            params['total'] = len(matches)

        if verbose >= 1:
            print(f'Matches before filtering: {len(matches)}')
        
        good_matches = self._lowes_test_ratio(matches, ratio=lowes_ratio)
        
        if verbose >= 2:
            print(f"Matches after Lowe's ratio test: {len(good_matches)}")
        
        if mutual:
            if verbose >= 2:
                print('Performing mutual filtering...')
            good_matches = self._mutual_filter(matcher, des1, des2, good_matches, lowes_ratio)
        
        if verbose >= 1:
            print(f'Matches after filtering: {len(good_matches)}')

        if return_params:
            params['filtered'] = len(good_matches)

        return good_matches, params
    
    def _match_find_bior(
        self,
        feature: cv2.Feature2D, matcher: cv2.DescriptorMatcher,
        bior1_pyr: list[np.ndarray], bior2_pyr: list[np.ndarray],
        bior1_mask: list[np.ndarray], bior2_mask: list[np.ndarray],
        levels: list[int] = [0],
        mutual: bool = False,
        lowes_ratio: float = 0.75,
        weight_function: callable = None,
        sample_prc: float = 0.2,
        random_seed: int = 42,
        verbose:int = 0,
        return_params=False
    ):
    
        weights = []
        ret_matches = []
        ret_kp1, ret_des1 = [], []
        ret_kp2, ret_des2 = [], []
        
        level_kp_des_params = None
        level_match_params = None
        if return_params:
            level_kp_des_params = []
            level_match_params = []

        for l in sorted(levels, reverse=True):
            if verbose >= 1:
                print(f'Processing level {l}')
            kp1, des1, kp2, des2, kp_des_params = self._find_scaled(
                feature, 
                bior1_pyr[0], bior1_pyr[l], bior1_mask[l], 
                bior2_pyr[0], bior2_pyr[l], bior2_mask[l],
                return_params=return_params
            )
            if return_params:
                kp_des_params['level'] = l
                level_kp_des_params.append(kp_des_params)
            
            matches, match_params = self._filter_keypoints_and_matches(
                matcher, des1, des2,
                lowes_ratio, mutual,
                verbose,
                return_params=return_params
            )
            if return_params:
                match_params['level'] = l
                level_match_params.append(match_params)
            
            loc_weights = None
            if weight_function is not None:
                loc_weights = [weight_function(l, kp1[m.queryIdx].response, kp2[m.trainIdx].response) for m in matches]
                weights.extend(loc_weights)

            ret_matches = self._merge_matches(ret_matches, matches, len(ret_kp1), len(ret_kp2))
            
            ret_kp1.extend(kp1)
            ret_kp2.extend(kp2)
            ret_des1.extend(des1)
            ret_des2.extend(des2)

        params = None
        if return_params:
            params = {
                'match': level_match_params,
                'kp_des': level_kp_des_params
            }

        if weight_function is not None:
            ret_matches, (ret_kp1, ret_des1), (ret_kp2, ret_des2), sample_params = self._sample_matches(
                ret_matches, 
                ret_kp1, ret_kp2,
                ret_des1, ret_des2,
                np.array(weights), 
                sample_prc, random_seed,
                verbose=verbose,
                return_params=return_params
            )
            params['sample'] = sample_params

        return ret_matches, (ret_kp1, ret_des1), (ret_kp2, ret_des2), params
    
    def _find_homography_he_bior(
        self,
        bior_he1, bior_he2,
        bior_mask1, bior_mask2,
        feature: cv2.Feature2D,
        matcher: cv2.DescriptorMatcher,
        
        levels: list[int] = [0],
        verbose:int = 0,

        filter__mutual: bool = False,
        filter__lowes_ratio: float = 0.75,
        filter__weight_function: callable = None,
        filter__sample_prc: float = 0.2,
        filter__random_seed: int = 42,

        homography__ransacReprojThreshold=4.0,
        homography__refineIters=10,

        return__params=False
    ):

        get_ch = lambda bior, ch: [bior[i][...,ch] for i in range(len(bior))]

        bior_hem1 = get_ch(bior_he1, 0)
        bior_hem2 = get_ch(bior_he2, 0)
        bior_eos1 = get_ch(bior_he1, 1)
        bior_eos2 =  get_ch(bior_he2, 1)

        bior_channels = [(bior_hem1, bior_hem2), (bior_eos1, bior_eos2)]
        
        matches = []
        kp1 = []
        kp2 = []
        
        q_offset = 0
        t_offset = 0

        params = None
        channels = ['hem', 'eos']
        if return__params:
            params = {}

        for i, (ch1, ch2) in enumerate(bior_channels):
            ch_matches, (ch_kp1, _), (ch_kp2, _), ch_params = self._match_find_bior(
                feature, matcher,
                ch1, ch2,
                bior_mask1, bior_mask2,
                
                levels=             levels,
                verbose=            verbose,
                mutual=             filter__mutual,
                lowes_ratio=        filter__lowes_ratio,
                weight_function=    filter__weight_function,
                sample_prc=         filter__sample_prc,
                random_seed=        filter__random_seed,

                return_params=      return__params
            )

            if return__params:
                params[channels[i]] = ch_params

            if i == 0:
                q_offset += len(ch_kp1)
                t_offset += len(ch_kp2)

            kp1.extend(ch_kp1)
            kp2.extend(ch_kp2)
            matches.append(ch_matches)

        matches = self._merge_matches(matches[0], matches[1], q_offset, t_offset)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,

            method=                 cv2.RANSAC, 
            ransacReprojThreshold=  homography__ransacReprojThreshold,
            refineIters=            homography__refineIters
        )
        if H is None:
            raise ValueError("Affine matrix is not valid")

        if return__params:
            mask = mask.ravel().astype(bool)
            inlier_matches = [m for i, m in enumerate(matches) if mask[i]]

            params['ransac'] = {'inline':inlier_matches}

            return H, params
        
        return H

    def _json_to_params(self, param_json):
        def remove_prefix(s, prefix):
            s_lower = s.lower()
            prefix_lower = prefix.lower()
            
            if s_lower.startswith(prefix_lower):
                rest = s[len(prefix):]
                rest = rest.lstrip('_')
                return rest
            return s

        def build_feature(params):
            features = {
                'AKAZE': cv2.AKAZE_create,
                'KAZE': cv2.KAZE_create,
                'BRISK': cv2.BRISK_create,
                'SIFT': cv2.SIFT_create,
            }
            
            feature_type = params['feature_type'].upper()
            
            if feature_type not in features.keys():
                raise ValueError('Unsupported feature: {}!'.format(feature_type))
            
            feature = features[feature_type]
            feature_params = {}
            for param, value in params.items():
                if param.upper().startswith(feature_type):
                    feature_params[remove_prefix(param, feature_type)] = value

            return feature(**feature_params)

        def build_matcher(params):
            matchers = ['FLANN', 'BF']
            matcher_type = params['matcher_type'].upper()

            if matcher_type not in matchers:
                raise ValueError('Unsupported matcher: {}!'.format(matcher_type))
            
            matcher_params = {}

            for param, value in params.items():
                if param.upper().startswith(matcher_type):
                    matcher_params[remove_prefix(param, matcher_type)] = value

            if matcher_type == 'FLANN':
                return cv2.FlannBasedMatcher({
                    'algorithm': matcher_params.pop('algorithm', 1),
                    'trees': matcher_params.pop('trees', 5)
                },{
                    'checks': matcher_params.pop('checks', 50)
                })
            
            return cv2.BFMatcher(**matcher_params)

        def choice_to_params(choice_params):
            feature = build_feature(choice_params)
            matcher = build_matcher(choice_params)
            
            return {
                'feature':                              feature,
                'matcher':                              matcher,
                'levels':                               choice_params['levels'],
                'filter__mutual':                       choice_params['filter__mutual'],
                'filter__lowes_ratio':                  choice_params['filter__lowes_ratio'],
                'filter__weight_function':              self._w_function[choice_params['filter__weight_function']],
                'filter__sample_prc':                   choice_params['filter__sample_prc'],
                'homography__ransacReprojThreshold':    choice_params['homography__ransacReprojThreshold'],
                'homography__refineIters':              choice_params['homography__refineIters'],
            }
        
        print(f'Loading parameters from file: {param_json} ...')

        with open(param_json, 'r', encoding='utf-8') as f:
            params = choice_to_params(json.load(f)['params'])

        return params

    def __init__(self, param_json):
        self.w_function = {
            'linear': self._linear,
            'power' : self._power 
        }
        
        self._hyper_params = self._json_to_params(param_json)
        pass
    
    def find_transform(self, tissue1: Tissue, tissue2: Tissue, ret_params: bool = False):
        bior1, masks1 = self._he_bior_pyramid(tissue1, 5)
        bior2, masks2 = self._he_bior_pyramid(tissue2, 5)

        if ret_params:
            H, params = self._find_homography_he_bior(
                bior2, bior1, 
                masks2, masks1, 
                verbose = 0,
                filter__random_seed = 0,
                return__params=ret_params,
                **self._hyper_params
            )

            return H, params
        else:
            H = self._find_homography_he_bior(
                bior2, bior1, 
                masks2, masks1, 
                verbose = 0,
                filter__random_seed = 0,
                return__params=ret_params,
                **self._hyper_params
            )

            return H

    def register(self, tissue1: Tissue, tissue2: Tissue, ret_params: bool = False):
        def warp_ch(img, shape, H):
            warp = lambda ch, shape, H: cv2.warpAffine(
                ch, H, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

            if len(img.shape) < 3:
                return warp(img, shape, H)

            return np.stack([warp(img[...,i], shape, H) for i in range(img.shape[2])])
        
        if ret_params:
            H, params = self.find_transform(tissue1, tissue2, ret_params)
        else:
            H = self.find_transform(tissue1, tissue2, ret_params)

        tissue_reg = Tissue(
            he = warp_ch(tissue2.he, tissue1.he.shape, H),
            rgb = warp_ch(tissue2.rgb, tissue1.rgb.shape, H),
            mask = warp_ch(tissue2.mask.astype(np.uint8), tissue1.mask.shape, H).astype(bool)
        )

        if ret_params:
            return tissue_reg, params
        else:
            return tissue_reg
        
class NonAffineRegister(Register):
    def _mmi_register(
        self,
        img1: np.ndarray, img2: np.ndarray,
        mask1: np.ndarray = None, mask2: np.ndarray = None,
        initial_transform: sitk.Transform = None,
        register = False,

        num_bins: int = 100,
        metric_sampling_strategy: str = "RANDOM",
        metric_sampling_percentage: float = 0.4,
        metric_use_gradient: bool = True,

        coarse_iterations: int = 150,
        coarse_tol: float = 1e-5,
        coarse_max_corrections: int = 5,
        coarse_max_func_evals: int = 300,
        coarse_cost_factor: float = 1e8,

        fine_iterations: int = 250,
        fine_tol: float = 1e-6,
        fine_max_corrections: int = 7,
        fine_max_func_evals: int = 500,
        fine_cost_factor: float = 1e9,
        fine_metric_sampling_percentage: float = 0.6,

        shrink_factors: list = [4, 2, 1],
        smoothing_sigmas: list = [2.5, 1.2, 0],

        coarse_grid_spacing: list = [10, 10],
        fine_grid_spacing: list = [32, 32],
        spline_order: int = 3,

        blur_sigma: float = 3.0
    ) -> tuple[sitk.Image, sitk.Transform]:
        img1_sitk = sitk.GetImageFromArray(img1.astype(np.float32))
        img2_sitk = sitk.GetImageFromArray(img2.astype(np.float32))
        img1_sitk.SetSpacing((1.0,) * img1_sitk.GetDimension())
        img2_sitk.SetSpacing((1.0,) * img2_sitk.GetDimension())
        img1_sitk.SetOrigin((0.0,) * img1_sitk.GetDimension())
        img2_sitk.SetOrigin((0.0,) * img2_sitk.GetDimension())

        mask1_sitk, mask2_sitk = None, None
        if mask1 is not None and mask2 is not None:
            mask1 = mask1.astype(np.uint8)
            mask2 = mask2.astype(np.uint8)
            kernel = np.ones((7, 7), np.uint8)
            mask1 = cv2.erode(mask1, kernel, iterations=1)
            mask2 = cv2.erode(mask2, kernel, iterations=1)
            mask1_sitk = sitk.GetImageFromArray(mask1.astype(np.uint8))
            mask2_sitk = sitk.GetImageFromArray(mask2.astype(np.uint8))
            mask1_sitk.CopyInformation(img1_sitk)
            mask2_sitk.CopyInformation(img2_sitk)

        def configure_registration(fixed, moving, fixed_mask, moving_mask, transform, 
                                sampling_percentage, iterations, tol, max_corr, max_eval, cost_factor):
            reg = sitk.ImageRegistrationMethod()

            reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=num_bins)
            reg.SetMetricSamplingStrategy(getattr(reg, metric_sampling_strategy))
            reg.SetMetricSamplingPercentage(sampling_percentage)
            reg.SetMetricUseFixedImageGradientFilter(metric_use_gradient)
            reg.SetMetricUseMovingImageGradientFilter(metric_use_gradient)

            if fixed_mask is not None:
                reg.SetMetricFixedMask(fixed_mask)
            if moving_mask is not None:
                reg.SetMetricMovingMask(moving_mask)

            reg.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=tol,
                numberOfIterations=iterations,
                maximumNumberOfCorrections=max_corr,
                maximumNumberOfFunctionEvaluations=max_eval,
                costFunctionConvergenceFactor=cost_factor
            )
            reg.SetOptimizerScalesFromPhysicalShift()
            reg.SetInterpolator(sitk.sitkBSpline)

            reg.SetShrinkFactorsPerLevel(shrink_factors)
            reg.SetSmoothingSigmasPerLevel(smoothing_sigmas)
            reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            reg.SetInitialTransform(transform, inPlace=False)
            return reg

        if initial_transform:
            img2_sitk = sitk.Resample(
                img2_sitk,
                img1_sitk,
                initial_transform,
                sitk.sitkBSpline
            )

        coarse_transform = sitk.BSplineTransformInitializer(
            img1_sitk,
            coarse_grid_spacing,
            order=spline_order
        )

        img1_blurred = sitk.DiscreteGaussian(img1_sitk, blur_sigma)
        img2_blurred = sitk.DiscreteGaussian(img2_sitk, blur_sigma)

        reg_coarse = configure_registration(
            img1_blurred,
            img2_blurred,
            mask1_sitk,
            mask2_sitk,
            coarse_transform,
            metric_sampling_percentage,
            coarse_iterations,
            coarse_tol,
            coarse_max_corrections,
            coarse_max_func_evals,
            coarse_cost_factor
        )
        tform_coarse = reg_coarse.Execute(img1_blurred, img2_blurred)

        img2_warped = sitk.Resample(
            img2_sitk,
            img1_sitk,
            tform_coarse,
            sitk.sitkBSpline
        )

        fine_transform = sitk.BSplineTransformInitializer(
            img1_sitk,
            fine_grid_spacing,
            order=spline_order
        )

        reg_fine = configure_registration(
            img1_sitk,
            img2_warped,
            mask1_sitk,
            mask2_sitk,
            fine_transform,
            fine_metric_sampling_percentage,
            fine_iterations,
            fine_tol,
            fine_max_corrections,
            fine_max_func_evals,
            fine_cost_factor
        )

        tform_fine = reg_fine.Execute(img1_sitk, img2_warped)

        final_transform = sitk.CompositeTransform(img1_sitk.GetDimension())
        if initial_transform:
            final_transform.AddTransform(initial_transform)
        final_transform.AddTransform(tform_coarse)
        final_transform.AddTransform(tform_fine)

        if register:
            reg_img = sitk.Resample(
                img2_sitk,
                img1_sitk,
                final_transform,
                sitk.sitkBSpline
            )

            return sitk.GetArrayFromImage(reg_img), final_transform
        return final_transform

    def _merge_pca(self, tissue):
        hem, eos = tissue.hem, tissue.eos

        flat = np.stack([hem.ravel(), eos.ravel()], axis=1)
        pca = PCA(n_components=1)
        merged = pca.fit_transform(flat).reshape(hem.shape)
        return merged

    def _json_to_params(self, param_json):
        print(f'Loading parameters from file: {param_json} ...')

        with open(param_json, 'r', encoding='utf-8') as f:
            params = json.load(f)['params']

        return params

    def __init__(self, param_json):
        self._hyperparams = self._json_to_params(param_json)
        pass

    def find_transform(self, tissue1: Tissue, tissue2: Tissue):
        return self._mmi_register(
            self._merge_pca(tissue1), self._merge_pca(tissue2),
            tissue1.mask, tissue2.mask,
            register=False,
            **self._hyperparams
        )
    
    def register(self, tissue1: Tissue, tissue2: Tissue):
        sitk_trans = self.find_transform(tissue1, tissue2)

        hem = tissue1.hem, tissue2.hem
        eos = tissue1.eos, tissue2.eos
        red = tissue1.rgb[0], tissue2.rgb[0]
        green = tissue1.rgb[1], tissue2.rgb[1]
        blue = tissue1.rgb[2], tissue2.rgb[2]
        mask = tissue1.mask.astype(np.uint8), tissue2.mask.astype(np.uint8)

        def transform(channel_list):
            result = []
            for ch1, ch2 in channel_list:
                dtype = ch2.dtype
                ch1 = sitk.GetImageFromArray(ch1)
                ch2 = sitk.GetImageFromArray(ch2)
                ch1.SetSpacing((1.0, 1.0))
                ch1.SetOrigin((0.0, 0.0))
                ch2.SetSpacing((1.0, 1.0))
                ch2.SetOrigin((0.0, 0.0))
                res = sitk.Resample(
                    ch2,
                    ch1,
                    sitk_trans,
                    sitk.sitkBSpline
                )
                result.append(sitk.GetArrayFromImage(res).astype(dtype))
            if len(result) == 1:
                return result[0]
            else:
                return np.stack(result, axis=-1)
        
        return Tissue(
            he = transform([hem, eos]),
            rgb = transform([red, green, blue]),
            mask = transform([mask]).astype(bool)
        )