# Multinomial Logit Model
# packages required
require('mlogit')
require('nnet')

#In built dataset
data()
Mdata <- read.csv("C:/RAVI/Data science/Assignments/Module 10 Multinomial Regression/Dataset/mdata.csv")
View(Mdata)
attach(Mdata)
head(Mdata)
table(Mdata$prog)
summary(Mdata)
Mdata1 <-  Mdata[ ,-c(1,2)]
View(Mdata1)

model <- multinom(prog ~ female + ses + schtyp + read + write + math + science + honors, data=Mdata1)
summary(model)

Mdata1$prog  <- relevel(Mdata1$prog, ref= "academic")  # change the baseline level


##### Significance of Regression Coefficients###
z <- summary(model)$coefficients / summary(model)$standard.errors
z
#2-tailed z test we get p values
p_value <- (1-pnorm(abs(z),0,1))*2

summary(model)$coefficients
p_value

# odds ratio 
exp(coef(model))

# predict probabilities
prob <- fitted(model)
prob

# Find the accuracy of the model

class(prob)
prob <- data.frame(prob)
View(prob)
prob["pred"] <- NULL

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

pred_name <- apply(prob,1,get_names)
?apply
prob$pred <- pred_name
View(prob)

# Confusion matrix
table(pred_name,Mdata1$prog)


# confusion matrix visualization
barplot(table(pred_name,Mdata1$prog),beside = T,col=c("red","lightgreen","blue"),legend=c("academic","general","vocation"),main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")


# Accuracy 
mean(pred_name==Mdata1$prog)

