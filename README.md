# Predator-Prey-Modeling-
Modeled the dynamics of Predator prey using UDE framework in Julia .

UNIVERSAL DIFFERENTIAL EQUATION FOR MODELING PREDATOR-PREY DYNAMICS.

 ● In this we tried to model only the interaction of Wolves and rabbits UDEs/NNs. It makes the problem/code more interesting! 
● In the following code, the model was trained only for a single predator-prey cycle, i.e., t = (0.0, 3.5). 
● The UDE model's extrapolation beyond this time span was also analyzed 
● The actual equation for the Dynamics are given below 

 dx /dt = αx − βxy

 dy /dt = −δy + γxy 

But we went one step further and replaced the interaction term ie (x*y) with a neural network , to do its magic, by magic i mean , even if we are not sure in future how the interaction will remain this model gives us the perfect solution to do so . 
dx /dt = αx − NNr

 dy /dt = −δy + NNw

 ● In the Fig below the representation is soo beautiful and accurate as the neural network is able to replicate the data for exact time span of (0-3.5) with the original data .
 ● This shows the power of neural network backed with the differential equation . 
● The scatter plot goes flat after ~t=3.5 because we did not train the model for the further time 
![Screenshot 2024-10-17 201304](https://github.com/user-attachments/assets/14db5c71-8d5a-45e7-a275-167dbf417222)
