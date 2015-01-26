sz = 384; % Size to cut out of rectangle image
N = 192; % downsampled
M = 60 * 150; % 90 seconds * 150fps = 9000   (Long)
%M = 30 * 150; % 30 seconds * 150fps = 4500  (Medium)
%M = 3  * 150; % 3  seconds * 150fps = 450   (Short)

[fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);

IMAGES = randn(N*N,M);

for i=1:M
    fprintf('trial %d\r', i)
    im_rgb=double(imread(num2str(i, 'png_2/q10-duck--%05d.png')));

    % Cut out sz by sz part of the image
    [rsz csz c] = size(im_rgb);
    im_rgb = im_rgb(rsz/2-sz/2+1 : rsz/2+sz/2, csz/2-sz/2+1 : csz/2+sz/2, :);
    im_rgb = im_rgb / 255.0;

    % Color to B/W
    im = rand(sz, sz);
    for r=1:sz
        for c=1:sz
            % CCIR Reccomends this: http://en.wikipedia.org/wiki/Luma_(video)#Rec._601_luma_versus_Rec._709_luma_coefficients
            im(r,c) = 0.2989 * im_rgb(r,c,1) + 0.5870 * im_rgb(r,c,2) + 0.1140 * im_rgb(r,c,3);
        end
    end

    If=fft2(im);

    % Down sample
    If=fftshift(If);
    If=If(sz/2-N/2+1 : sz/2+N/2, sz/2-N/2+1 : sz/2+N/2);
    If=ifftshift(If);

    imagew=real(ifft2(If.*fftshift(filt)));
    IMAGES(:,i)=reshape(imagew,N*N,1);
end

IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));

IMAGES_DUCK_LONG = randn(N,N,M);
for i=1:M
    IMAGES_DUCK_LONG(:,:,i) = reshape(IMAGES(:,i), N,N,1);
end

save IMAGES_DUCK_LONG.mat IMAGES_DUCK_LONG;
