#pointer on any function and then F1

library(h2o)

h2o.init(max_mem_size = "600M")

train.hex<-h2o.importFile("data/ames_train.csv")

h2o.describe(train.hex)

test.hex <- h2o.importFile("data/ames_test.csv")
#train <- read.csv("data/ames_test.csv")
#train.hex <- as.h2o(train)

h2o.describe(test.hex)
h2o.nrow(test.hex)


#AUTO ML
y <- "SalePrice"
#x <- h2o.names(train.hex)
x <- setdiff(h2o.names(train.hex), c("Id",y))

mod <- h2o.automl(x=x, y=y, training_frame = train.hex, 
                  max_models = 10, seed = 42)
mod

lb <- mod@leaderboard
print(lb, n=nrow(lb))

typeof(lb)
#to move to your local memory as a dataframe
lbdf <- as.data.frame(lb)

allmod <- grep("AllModels", lbdf$model_id, value ="TRUE")

se_am <- h2o.getModel(allmod)

se_am_ml <- h2o.getModel(se_am@model$metalearner$name)
                          
h2o.varimp(se_am_ml)
h2o.varimp_plot(se_am_ml)

best_mod <- h2o.getModel(lbdf$model_id[1])
h2o.varimp_plot(best_mod)

# Prediction
best_pred <- h2o.predict(mod@leader,newdata = test.hex)
write.csv(as.data.frame(best_pred), 
          "best_mod_pred_1.csv", 
          row.names = FALSE)

# Assignment
xrt_pred <- h2o.predict(h2o.getModel(lbdf$model_id[9]), 
                        newdata = test.hex)

plot(as.data.frame(xrt_pred)$predict,
as.data.frame(best_pred)$predict)

h2o.shutdown()

### Classification
h2o.init(max_mem_size = "600M")

wine_red <- read.csv("data/winequality-red.csv", 
                     stringsAsFactors = FALSE,
                     sep = ";")
str(wine_red)

wine_white <- read.csv("data/winequality-white.csv", 
                       stringsAsFactors = FALSE,
                       sep = ";")
str(wine_white)

# Check if variable names are the same
all.equal(names(wine_red), names(wine_white))

# Adding feature
wine_red$type <- "red"
wine_white$type <- "white"

# Row binding, inspecting
wine <- rbind(wine_red, wine_white)
str(wine)
hist(wine$quality)

wine$quality_class <- factor(as.integer(wine$quality >= 7), levels = c(0, 1), labels = c("avg", "premium"))

# To h2o
wine.hex <- as.h2o(wine)
y <- "quality_class"
X <- setdiff(h2o.names(wine.hex), c("quality", y))

wine_split <- h2o.splitFrame(data = wine.hex)

wine_model <- h2o.automl(X, y,
                         training_frame = wine_split[[1]],
                         max_models = 10,
                         seed = 84)

# Getting models leaderboard
lb <- wine_model@leaderboard
print(lb, n = nrow(lb))

lb_df <- as.data.frame(wine_model@leaderboard)

# Getting the best model that is not a stacked ensemble, and checking variable importance
h2o.varimp_plot(h2o.getModel(lb_df$model_id[3]))

top_model <- h2o.getModel(lb_df$model_id[3])
h2o.performance(top_model,
                newdata = wine_split[[2]])

