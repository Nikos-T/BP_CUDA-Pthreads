uslaeps = [100;200;400;800;1600;3200;6400;12800;25600;51200; 102400; 204800; 409600];
threads = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
times = zeros(length(uslaeps), length(threads));
fores = zeros(length(uslaeps), length(threads));
fp = fopen('test.txt');
tline = fgetl(fp);
while ischar(tline)
    [thrInd, uwaitInd, time] = strread(tline, 'threads = %u, wait = %u, time = %u.0');
    thrInd = log2(thrInd)-1;
    uwaitInd = log2(uwaitInd/100)+1;
    times(uwaitInd, thrInd) = times(uwaitInd,thrInd) + time;
    fores(uwaitInd, thrInd) = fores(uwaitInd,thrInd) + 1;
    tline = fgetl(fp);
end
times = round(times./fores);
fclose(fp);