## Question 1
 We want to minimize the distance between the our prediction and the known values. This is done with a normal regression.
 
 $$ min_{A12, A21} \sum_{n=1} ^{11}\left(P\left(Xi,A\right)-P_{given}\right)^2$$
 
 Then this is solved using a gradient decent algorithm as is seen in the code. This give us the solution that:
Regression estimation A12 and A21 is: 1.9110, 1.7293, with the regression final loss at:  0.87836564

We can see this  answer is quite acurate when compared to the real data, as shown in the graph

![image](https://user-images.githubusercontent.com/73143081/194738405-90d88e00-947f-4cb5-97f8-6ed15ee88cc3.png)
