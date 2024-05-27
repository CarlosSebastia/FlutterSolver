%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This file is part of jAERO Software 
%
%   THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
%   WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
%   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
%   NO EVENT SHALL THE COPYRIGHT OWNER BE LIABLE FOR ANY DIRECT, INDIRECT,
%   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
%   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
%   OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
%   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
%   TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
%   USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
%   DAMAGE.
%
%   Copyright (C) 2018 by Jan Schwochow (janschwochow@web.de)
%   $Revision: 2.0 $  $Date: 2018/09/23 12:00:00 $
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hf,V,freq,damp,vcrit,fcrit,mcrit] = vgfread(FileName,isel)
% reading NASTRAN95 out-file to extract flutter damping and frequency
% curves
%

if nargin < 2 
    isel = [];
end

% read f06-file
data = readfile(FileName);

ldata = length(data);
kk = [];
mm = [];
for ii = 1:ldata
   ind = strfind(data{ii},'FLUTTER  SUMMARY');
   if ~isempty(ind)
       disp([num2str(ii) data{ii}])
       kk = [kk ii+6];
       mm = [mm ii-4];
   end
end
mm(1) = [];
mm(end+1) = mm(end)-mm(end-1)+mm(end);

k = [];
V = [];
g = [];
f = [];
for ii = length(kk):-1:1
    nn = 0;
    for jj = kk(ii):mm(ii)
        nn = nn+1;
        ind = strfind(data{jj},'********');
        if ~isempty(ind)
            data{jj}(ind:ind+7) =  ' 1.0E+12';
        end
        num = str2num(data{jj});
        k(nn,ii) = num(1);
        V(nn,ii) = num(3);
        g(nn,ii) = num(4);
        f(nn,ii) = num(5);
    end
end

om = 2*pi*f;
lam = g.*om + 1i*om;
%lam = rootshuffle2(lam);
f = imag(lam)/2/pi;
g = real(lam)./imag(lam);

if ~isempty(isel) 
  V = V(:,isel);
  g = g(:,isel);
  f = f(:,isel);
  k = k(:,isel);
end

V = V(:,1);
damp = -g*100/2;
freq = f;
% find critical flutter speeds
vcrit = [];
fcrit = [];
mcrit = [];
kk = 0;
for ii = 1:size(damp,2)
    ind = find(sign(damp(:,ii)) < 0);
    if ~isempty(ind)
        ind = ind(1);
        if ind == 1
            Vi = V(1);
        else
            Vi = (V(ind)+V(ind-1))/2;
        end
        kk = kk+1;
        ind = ind(1);
        vcrit(kk) = interp1(V,V,Vi,'linear');
        fcrit(kk) = interp1(V,freq(:,ii),Vi,'linear');
        mcrit(kk) = ii;  
    end
end
[vcrit,ind] = sort(vcrit,'ascend');
fcrit = abs(fcrit(ind));
mcrit = mcrit(ind);
freq0 = freq(1,mcrit);

% print out summary
disp('flutter summary')
disp(['mode_crit  V_crit      freq_crit      freq_0'])
for ii = 1:length(mcrit)
    disp([num2str([mcrit(ii) vcrit(ii) fcrit(ii) freq0(ii)],'%12.2f')])
end

% flutter plot
hf = figure();
subplot(2,1,2)
plot(V,freq,'-');
xlim([V(1,1) V(end,1)])
ylim([0 30])
ylabel('Frequency') 
grid on
grid minor
xlabel('Speed')
subplot(2,1,1)
plot(V,damp,'-');  
xlim([V(1,1) V(end,1)])
ylim([-2 12])
ylabel('Damping') 
grid on
grid minor
title(FileName,'interpreter','none')

end


