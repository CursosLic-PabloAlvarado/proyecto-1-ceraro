## Copyright (C) 2021 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5852 IntroducciÃ³n al Reconocimiento de Patrones
## Escuela de IngenierÃ­a ElectrÃ³nica
## TecnolÃ³gico de Costa Rica

## Ejemplo de configuraciÃ³n de red neuronal y su entrenamiento

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

reuseNetwork = false; # Con reusenetwork = true la red una vez iniciado el entrenamiento, irá almacenando la red en un archivo 

if (reuseNetwork && exist(file,"file")==2) # Si encuentra ese archivo  
  ann.load(file); # Entonces cárguelo para que nadamás siga entrenando donde quedó
else # Aqui estoy formando una red 
  ann.nEpochs=500;
  ann.alpha=0.0001;  ## Learning rate
  ann.beta2=0.99;  ## ADAM si beta2>0
  ann.beta=0.9;    ## Momentum
  ann.minibatch=32;
  ann.method="stochastic";
  
  # Se está usando un cell arrays (arreglo de celdas) {}. Una celda es cualquier cosa
  ann.add({input_layer(2), # Capa de entrada que recibe 2 dimensiones
           #batchnorm(),
           dense_unbiased(16), # Capa densa sin sesgo 
           PReLU(), #
           #batchnorm(), 
           dense_unbiased(16),
           PReLU(),
           #batchnorm(),
           dense_unbiased(numClasses),
           SoftMax()});
  ann.add(EC()); # Capa de pérdida
endif

loss=ann.train(X,Y,vX,vY); # Se entrena la red 
ann.save(file); 

## TODO: falta agregar el resto de pruebas y visualizaciones