pkg load control
num = [0 0 10];
den = [0 1 10];
t = 0 : 0.02 : 10;
sys = tf(num,den);
step(sys,t);
grid 
title('Funcion escalon')