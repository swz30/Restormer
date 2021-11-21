close all;clear all;

denoised = load('./results/Real_Denoising/SIDD/mat/Idenoised.mat');
gt = load('./Datasets/test/SIDD/ValidationGtBlocksSrgb.mat');

denoised = denoised.Idenoised;
gt = gt.ValidationGtBlocksSrgb;
gt = im2single(gt);

total_psnr = 0;
total_ssim = 0;
for i = 1:40
    for k = 1:32
       denoised_patch = squeeze(denoised(i,k,:,:,:));
       gt_patch = squeeze(gt(i,k,:,:,:));
       ssim_val = ssim(denoised_patch, gt_patch);
       psnr_val = psnr(denoised_patch, gt_patch);
       total_ssim = total_ssim + ssim_val;
       total_psnr = total_psnr + psnr_val;
   end
end
qm_psnr = total_psnr / (40*32);
qm_ssim = total_ssim / (40*32);

fprintf('PSNR: %f SSIM: %f\n', qm_psnr, qm_ssim);

