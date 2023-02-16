close all
clear all
clc

%% training and testing data

snb_data=readtable("snb_normdata"); % odczytanie danych z pliku xls

x=table2array(snb_data(1:400,1:15)); % zamiana tablicy z danymi na macierz
t=table2array(snb_data(1:400,16));   % zamiana tablicy z informacją, czy pacjent jest chory (ckd/notckd) na wektor

xtrain=[x(1:200,:); x(251:370,:)];   % dane treningowe - 200 wektorów dla pacjentów chorych i 120 dla pacjentów zdrowych
xtrain=[xtrain ones(320,1)];		%dodatkowa "zmienna" zawsze rowna 1 (odpowiedniek polaryzacji i wagi w0) 

xtest=[x(201:250,:); x(371:400,:)];  % dane testowe - 50 wektorów dla pacjentów chorych i 30 dla pacjentów zdrowych
xtest=[xtest ones(80,1)];

ttrain=[t(1:200,:); t(251:370,:)];   % dane treningowe - 200 informacji, że pacjent jest chory i 120, że jest zdrowy - kompatybilne z macierzą xtrain
ttest=[t(201:250,:); t(371:400,:)] ; % dane testowe - 50 informacji, że pacjent jest chory i 30, że jest zdrowy - kompatybilne z macierzą ttrain

%% learning
%w = (rand(4,15)*10-1)/100;          % inicjacja macierzy w z niewielkimi początkowymi losowymi wartościami wag
w = (rand(5,16)*10-1)/100;          % inicjacja macierzy w z niewielkimi początkowymi losowymi wartościami wag (o jedna wage wiecej, aby uwzglednic polaryzacje)

%omega = (rand(1,4)*10-1)/100;        % inicjacja wektora omega z niewielkimi początkowymi losowymi wartościami wag
omega = (rand(1,5)*10-1)/100;        % inicjacja wektora omega z niewielkimi początkowymi losowymi wartościami wag

epochs=1000;		%liczba epok

eta = 0.001;   % współczynnik uczenia sieci

nt=size(xtrain,1);

for e=1:epochs
	index=randperm(nt);
	for k=1:nt
		
		i = index(k);  %wybór losowego wektora uczącego (z wektorów treningowych)

		ni = w * xtrain(i,:)';  % obliczenie pobudzenia neuronów warstwy ukrytej

		ksi = sigmoid_unipolar(ni); % obliczenie stanu wyjść neuronów warstwy ukrytej

		v = omega * ksi; % obliczenie pobudzenia neuronów warstwy wyjściowej

		y = sigmoid_unipolar(v);    % obliczenie stanu wyjść neuronów warstwy wyjściowej

		delta_o = (ttrain(i)-y) * derivative(v); % obliczenie sygnału błędu dla warstwy wyjściowej

		delta_h = derivative(ni) .* (omega * delta_o)'; % obliczenie sygnału błędu dla warstwy ukrytej

		omega = omega + eta * delta_o * ksi';   % zmodyfikowanie wagi warstwy wyjściowej

		w = w + eta * delta_h * xtrain(i,:);   % zmodyfikowanie wagi warstwy ukrytej

		E(k) = 0.5 * (ttrain(i)-y)^2;    % obliczenie poziomu błędu
	end

end
%% testing

for l=1:80
    ni = w * xtest(l,:)';  % pobudzenia neuronów warstwy ukrytej

    ksi = sigmoid_unipolar(ni); % stan wyjść neuronów warstwy ukrytej

    v = omega * ksi; % pobudzenia neuronów warstwy wyjściowej

    y(l) = prog(sigmoid_unipolar(v)); % stan wyjść neuronów warstwy wyjściowej

   
end

%% plots

plot(y,'g*');
hold on;
plot(ttest,'c--');
hold off;
title('Proces uczenia sieci - test');
ylabel('ckd = 0 / notckd = 1');
xlabel('Liczba iteracji');
legend('Wartosci testowe', 'Wartosci prawdziwe','Location','northwest');

%% statistics
TPR=(50-sum(y(1:50)))/50 % czułość

SPC=sum(y(51:80))/30 % specyficzność

%% Wykres błędu
blad=figure();
plot(1:k, E(1:k));
set(blad,'Position',[1 1 2000 1000]);
title('Przebieg bledu w zaleznosci od liczby iteracji');
xlabel('Liczba iteracji');
ylabel('Wartosc bledu');

%% functions
function y = prog(x) % funkcja progowania
   if x>0.5
       y=1;
   else
       y=0;
   end  
   
end
function y = sigmoid_unipolar(x) % funckja aktywacji - sigmoidalna unipolarna
   beta = 10;
   y = 1./(1+exp(-beta*(x)));
end

function y = derivative(x) %pochodna funkcji aktywacji
   beta = 10;
   y=(beta *exp(-beta*(x)))./(1 + exp(-beta*(x))).^2;
end