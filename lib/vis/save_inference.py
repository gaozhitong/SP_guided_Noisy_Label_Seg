import os
import numpy as np
import skimage.io
import nibabel as nib
import ipdb

def save_pred(data_list, names=None, id=None, title='tmp', phase='val', save_type=None,
              label_overlay=False):
    '''
    label_overlay: pred & gt label vis, or label on image.
                   requires:  ['pred', 'image', 'gt_label']
                   if label_overlay = True, we do not save 'image' and 'gt_label' by default.
    '''

    def make_dir(dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    save_dir = os.path.join(os.getcwd(), 'output/%s_%s' % (title, phase))
    make_dir(save_dir)

    if not label_overlay:
        for name, data in zip(names, data_list):
            _dir = os.path.join(save_dir, name)
            make_dir(_dir)
            png_prefix = _dir + '/%s' % str(id)

            data = data.numpy()
            if (name in ['pred', 'gt_label']):
                data = data.astype(np.uint8)
            B = data.shape[0]

            if save_type == '2d':
                for ith in range(B):
                    save_path = png_prefix + '.png'
                    output = data[ith] if name != 'image' else data[ith, 0]
                    if name in ['pred', 'gt_label']:
                        output = (output * 255).astype(np.uint8)
                        skimage.io.imsave(save_path, output, check_contrast=False)
                    elif name in ['image', 'heatmap']:
                        output = (output * 255.).astype(np.uint8)
                        skimage.io.imsave(save_path, output, check_contrast=False)
                    elif name.split('_')[-1]  == 'npy':
                        save_path = png_prefix + '.npy'
                        with open(save_path, 'wb') as f:
                            np.save(f, output)

            elif save_type == '3d':
                # .nii.gz
                for ith in range(B):
                    save_path = png_prefix + '.nii.gz'  # _'%02d.nii.gz' % ith
                    output = data[ith] if name != 'image' else data[ith, 0]

                    image = nib.Nifti1Image(output, np.eye(4))
                    nib.save(image, save_path)
    else:
        # ['pred', 'image', 'gt_label']
        data = {
            'pred': data_list[0].numpy().astype(np.uint8),
            'image': data_list[1].numpy(),
            'gt_label': data_list[2].numpy().astype(np.uint8),
        }
        B = data['pred'].shape[0]
        _dir = os.path.join(save_dir, 'label_overlay')
        make_dir(_dir)
        png_prefix = _dir + '/%s' % str(id)

        _dir2 = os.path.join(save_dir, 'image_overlay')
        make_dir(_dir2)
        png_prefix2 = _dir2 + '/%s' % str(id)

        if save_type == '2d':
            for ith in range(B):
                # overlay label: TP Green, FP Red, FN blue
                save_path = png_prefix + '.png'
                pred = data['pred'][ith]
                image = data['image'][ith]; image = np.transpose(image, (1,2,0))
                gt_label = data['gt_label'][ith]

                # TP 1, FP 2, FN 3
                label_ = np.zeros_like(pred)
                label_[np.logical_and(pred > 0, gt_label > 0)] = 1
                label_[np.logical_and(pred > 0, gt_label == 0)] = 2
                label_[np.logical_and(pred == 0, gt_label > 0)] = 3
                output = np.zeros_like(image, dtype=np.uint8)
                output[:, :, 1][label_ == 1] = 255  # G
                output[:, :, 0][label_ == 2] = 255  # R
                output[:, :, 2][label_ == 3] = 255  # B
                skimage.io.imsave(save_path, output, check_contrast=False)

                # overlay image
                save_path2 = png_prefix2 + '.png'
                image = ((image + 1.0) / 2 * 255).astype(np.uint8)
                iweight = 0.8
                output2 = image * iweight + output * (1-iweight)
                output2 = output2.astype(np.uint8)
                skimage.io.imsave(save_path2, output2, check_contrast=False)

