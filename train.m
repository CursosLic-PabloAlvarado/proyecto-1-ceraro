## Copyright (C) 2021 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5852 Introducción al Reconocimiento de Patrones
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## Ejemplo de configuración de red neuronal y su entrenamiento

1;
pkg load statistics;

numClasses=4;

##datashape='spirals';
##datashape='curved';
datashape='vertical';


[X,Y]=create_data(numClasses*50,numClasses,datashape);   ## Training
[vX,vY]=create_data(numClasses*40,numClasses,datashape); ## Validation
figure(1,"name","Datos de entrenamiento");
hold off;
plot_data(X,Y);

ann=sequential();

file="ann.dat";

reuseNetwork = false;

if (reuseNetwork && exist(file,"file")==2)
  ann.load(file);
else
  ann.nEpochs=500;
  ann.alpha=0.01;  ## Learning rate
  ann.beta2=0.99;  ## ADAM si beta2>0
  ann.beta=0.9;    ## Momentum
  ann.minibatch=32;
  ann.method="stochastic";

  ann.add({input_layer(2),
           dense_unbiased(16),
           sigmoide(),
           dense_unbiased(16),
           sigmoide(),
           dense_unbiased(numClasses),
           sigmoide()});
  
  ann.add(olsloss());
endif

loss=ann.train(X,Y,vX,vY);
ann.save(file);

## TODO: falta agregar el resto de pruebas y visualizaciones