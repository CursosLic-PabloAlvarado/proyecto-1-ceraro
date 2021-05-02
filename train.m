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

reuseNetwork = false; # Con reusenetwork = true la red una vez iniciado el entrenamiento, ir� almacenando la red en un archivo 

if (reuseNetwork && exist(file,"file")==2) # Si encuentra ese archivo  
  ann.load(file); # Entonces c�rguelo para que nadam�s siga entrenando donde qued�
else # Aqui estoy formando una red 
  ann.nEpochs=500;
  ann.alpha=0.0001;  ## Learning rate
  ann.beta2=0.99;  ## ADAM si beta2>0
  ann.beta=0.9;    ## Momentum
  ann.minibatch=32;
  ann.method="stochastic";
  
  # Se est� usando un cell arrays (arreglo de celdas) {}. Una celda es cualquier cosa
  ann.add({input_layer(2), # Capa de entrada que recibe 2 dimensiones
           batchnorm(),
           dense_unbiased(numClasses), # Capa densa sin sesgo 
           sigmoide(),
           #PReLU(), #
           #batchnorm(), 
           #dense_unbiased(16),
           #PReLU(),
           #batchnorm(),
           #dense_unbiased(numClasses),
           #sigmoide()
           #SoftMax()
           });
  ann.add(MSE()); # Capa de p�rdida
endif

loss=ann.train(X,Y,vX,vY); # Se entrena la red 
ann.save(file); 

## TODO: falta agregar el resto de pruebas y visualizaciones

#x=linspace(-1,1,256);
#[GX, GY]=meshgrid(x,x);
#FX=[ones(size(GX(:))) GX(:) GY(:)];
#FZ=