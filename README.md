# Pix2Pix para reconstrucción de hologramas de Fresnel simulados computacionalmente.

La red neuronal generativa (GAN) fue puesta en la tarea de resolver la propagación inversa de un campo óptico simulado. Se encontró que con información parcial del campo óptico la GAN produjo resultados aceptables en el proceso de
reconstrucción. Esto representa una alternativa eficiente frente a los algoritmos clásicos de reconstrucción con la misma cantidad de información.

## Problema.

La holografía digital puede ser simulada mediante méto-
dos basados en transformada de Fourier (FFT)–discreta en
el caso computacional–. Para la generación de hologramas de Fresnel, de acuerdo a la teoría de difracción de Fresnel, se debe definir una expresión
para la propagación a cierta distancia del campo difractado
de un objeto (propagador de Fresnel). Para ello se calcula la
convolución del campo de entrada (onda transmitida por el
objeto) mediante una función de transferencia en el plano de
observación.  Para la reconstrucción del holograma se debe implementar la propa-
gación inversa, usando la función de transferencia de Fresnel
inversa $H_{-z}$ , por lo cual es necesario conocer la distancia $z$ de
propagación. El objetivo de este trabajo fue el diseño de un modelo GAN entrenado para revolver el problema del proceso de reconstrucción de hologramas de Fresnel.

La implementación está basada en el ejemplo de [GAN de TensorFlow](https://www.tensorflow.org/tutorials/generative/pix2pix)
