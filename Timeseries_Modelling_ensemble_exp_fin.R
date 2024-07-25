# Time Series Modelling ----
# 
# Featured: walmart_sales_weekly dataset
# multiple product lines, timeseries

# Libraries ----

library(modeltime)
library(tidymodels)
library(timetk)
library(tidyverse)
#library(modeltime.h2o)
library(lubridate)
library(xgboost)
library(parsnip)
library(dplyr)
library(modeltime.ensemble)

#remotes::install_github("business-science/modeltime.ensemble")
#install.packages("modeltime.ensemble")

# DATA ----
# - Time Series Visualization

data_tbl <- walmart_sales_weekly

data_tbl %>%
    group_by(id) %>%
    plot_time_series(
        .date_var = Date, 
        .value = Weekly_Sales, 
        .facet_ncol = 2,
        .smooth      = F,
        .interactive = F
        )

# Time Series decomposition:
data_tbl %>% 
    group_by(id) %>%
    plot_stl_diagnostics(.date_var = Date, .value = Weekly_Sales)
# freq = 13 observations per quarter, trend = 52 observations per year

# Time Series Splits ----
# - first nest the timeseries as a tibble-dataframe of timeseries
# - add 90 days of empty timeseries to the end of each -  this is the predicted
# - take last 90 days of actual data as testing data

# data_nested <- walmart_sales_weekly %>%
#     select(id, Date, Weekly_Sales) %>%
#     nest(nested_column = -id)
# 
# data_nested$nested_column

nested_data_tbl <- data_tbl %>%
    group_by(id) %>%
    extend_timeseries(    # adds in future timestamps, where the data will be missing
        .id_var = id,
        .date_var = Date,
        .length_future = 90
    ) %>%
    nest_timeseries(      # adds nested dataframe, value and date as actual and future
        .id_var = id,
        .length_future = 90
    ) %>%
    split_nested_timeseries(     # creates train and test splits on actual data
        .length_test = 90   # 90 days is the length of the test data set
    )

# test <- extract_nested_test_split(nested_data_tbl)
# train <- extract_nested_train_split(nested_data_tbl)



# 1.0 ARIMA Modelling ----

#rec_arima <- recipe(Weekly_Sales ~ ., extract_nested_train_split(nested_data_tbl)) %>%
    
    
## Auto Arima ----

#rec_arima <- recipe(Weekly_Sales ~ ., extract_nested_train_split(nested_data_tbl)) %>%
    
model_fit_arima <- arima_reg(seasonal_period = 52) %>%
    set_engine("auto_arima") %>%
    fit(Weekly_Sales ~ Date, extract_nested_train_split(nested_data_tbl))

model_fit_arima


## boosted ARIMA ----
model_fit_arima_boosted_1 <- arima_boost(
    seasonal_period = 52,
    min_n = 2,
    learn_rate = 0.0015
) %>%
    set_engine(engine = "auto_arima_xgboost") %>%
    fit(Weekly_Sales ~ Date + as.numeric(Date) + factor(months(Date)),
        data = extract_nested_train_split(nested_data_tbl))


model_fit_arima_boosted_1

# boosted arima 2 ----
model_fit_arima_boosted_2 <- arima_boost(
    seasonal_period = 52,
    min_n = 2,
    learn_rate = 0.015
) %>%
    set_engine(engine = "auto_arima_xgboost") %>%
    fit(Weekly_Sales ~ Date + as.numeric(Date) + factor(months(Date)),
        data = extract_nested_train_split(nested_data_tbl))


model_fit_arima_boosted_2

# boosted arima 3 ----
model_fit_arima_boosted_3 <- arima_boost(
    seasonal_period = 52,
    min_n = 2,
    learn_rate = 0.150
) %>%
    set_engine(engine = "auto_arima_xgboost") %>%
    fit(Weekly_Sales ~ Date + as.numeric(Date) + factor(months(Date)),
        data = extract_nested_train_split(nested_data_tbl))


model_fit_arima_boosted_3


## ETS Model ----
model_fit_ets <- exp_smoothing() %>%
    set_engine(engine = "ets") %>%
    fit(Weekly_Sales ~ Date, data = extract_nested_train_split(nested_data_tbl))


## Prophet Model ----
model_fit_prophet <- prophet_reg(
    seasonality_yearly = TRUE
) %>%
    set_engine(engine = "prophet") %>%
    fit(Weekly_Sales ~ Date, data = extract_nested_train_split(nested_data_tbl))

## LM Model ----
model_fit_lm <- linear_reg() %>%
    set_engine("lm") %>%
    fit(Weekly_Sales ~ as.numeric(Date) + factor(months(Date), ordered = FALSE),
        data = extract_nested_train_split(nested_data_tbl))

## MARS: Multivariate Adaptive Regression Spline ----
model_spec_mars <- mars(mode = "regression") %>%
    set_engine("earth") 

recipe_spec_mars <- recipe(Weekly_Sales ~ Date, data = extract_nested_train_split(nested_data_tbl)) %>%
    step_date(Date, features = "month", ordinal = FALSE) %>%
    step_mutate(date_num = as.numeric(Date)) %>%
    step_normalize(date_num) %>%
    step_rm(Date)

wflw_fit_mars <- workflow() %>%
    add_recipe(recipe_spec_mars) %>%
    add_model(model_spec_mars) %>%
    fit(extract_nested_train_split(nested_data_tbl))

## XGBoost ----
rec_xgb <- recipe(Weekly_Sales ~ ., extract_nested_train_split(nested_data_tbl)) %>%
    step_timeseries_signature(Date) %>%    # creates calenader features
    step_rm(Date) %>%                      # removes date feature for xgboost
    step_zv(all_predictors()) %>%           # removes zero variance predictors
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%   # dummy any nominal predictors
    step_integer(all_logical_predictors())

wflw_fit_xgb_1 <- workflow() %>%
    add_model(boost_tree("regression", learn_rate = 0.50) %>% set_engine("xgboost")) %>%
    add_recipe(rec_xgb)

wflw_fit_xgb_2 <- workflow() %>%
    add_model(boost_tree("regression", learn_rate = 0.35) %>% set_engine("xgboost")) %>%
    add_recipe(rec_xgb)

wflw_fit_xgb_3 <- workflow() %>%
    add_model(boost_tree("regression", learn_rate = 0.15) %>% set_engine("xgboost")) %>%
    add_recipe(rec_xgb)

wflw_fit_xgb_4 <- workflow() %>%
    add_model(boost_tree("regression", learn_rate = 0.05) %>% set_engine("xgboost")) %>%
    add_recipe(rec_xgb)

## THIEF - Temporal Hierarchical Forecasting ----
# rec_thief <- recipe(Weekly_Sales ~ Date, extract_nested_train_split(nested_data_tbl)) %>%
#     step_rm(-Date, -Weekly_Sales)
# 
# wflw_fit_thief <- workflow() %>%
#     add_model(temporal_hierarchy() %>% set_engine("thief")) %>%
#     add_recipe(rec_thief)

## Machine Learning - GLM ----
model_fit_glmnet_1 <- linear_reg(penalty = 0.01) %>%
    set_engine("glmnet") %>%
    fit(
        Weekly_Sales ~ as.numeric(Date) 
        + factor(months(Date), ordered = FALSE),
        extract_nested_train_split(nested_data_tbl)
    )

model_fit_glmnet_1


# GLM 2
model_fit_glmnet_2 <- linear_reg(penalty = 0.05) %>%
    set_engine("glmnet") %>%
    fit(
        Weekly_Sales ~ as.numeric(Date) 
        + factor(months(Date), ordered = FALSE),
        extract_nested_train_split(nested_data_tbl)
    )

model_fit_glmnet_2

# GLM 3
model_fit_glmnet_3 <- linear_reg(penalty = 0.1) %>%
    set_engine("glmnet") %>%
    fit(
        Weekly_Sales ~ as.numeric(Date) 
        + factor(months(Date), ordered = FALSE),
        extract_nested_train_split(nested_data_tbl)
    )

model_fit_glmnet_3

# # Tune the Models ----
# # time series cross validation
# set.seed(123)
# resample_spec <- time_series_cv(
#     extract_nested_train_split(nested_data_tbl),
#     date_var = Date,
#     initial = 10,
#     assess = 5,
#     skip = 5,
#     lag = 0,
#     cumulative = FALSE,
#     slice_limit = n(),
#     point_forecast = FALSE
# )
# 
# resample_spec %>% plot_time_series_cv_plan(Date, Weekly_Sales, .interactive = FALSE)



##################
# Model Fit ----
#################

parallel_start(6)  # for 6 clusters (i.e. cores)

nested_modeltime_tbl <- nested_data_tbl %>%
    #  slice_tail(n = 6) %>% # gives the last 6 (demonstration reasons)
    modeltime_nested_fit(
        
        model_list = list(
            model_fit_arima,
            model_fit_arima_boosted_1,
            model_fit_arima_boosted_2,
            model_fit_arima_boosted_3,
            model_fit_ets,
            model_fit_prophet,
            model_fit_lm,
            wflw_fit_mars,
            wflw_fit_xgb_1,
            wflw_fit_xgb_2,
            wflw_fit_xgb_3,
            wflw_fit_xgb_4,
            #wflw_fit_thief,
            model_fit_glmnet_1,
            model_fit_glmnet_2,
            model_fit_glmnet_3
            
        ),
        control = control_nested_fit(
            verbose = TRUE,
            allow_par = TRUE
        )
    )


nested_modeltime_tbl

#accuracy to select for models
nested_modeltime_tbl %>%
    extract_nested_test_accuracy() %>%
    table_modeltime_accuracy()


## Add Ensembles to the nested modeltime tbl ----

### Ensemble 1: mean of all models ----
ensem <- nested_modeltime_tbl %>%
    ensemble_nested_average(
        type           = "mean",
        keep_submodels = TRUE,
        model_ids      = c(4, 5, 8, 9, 10), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem %>% extract_nested_modeltime_table()

### Ensemble 2: median of all models ---
ensem_2 <- ensem %>%
    ensemble_nested_average(
        type           = "median",
        keep_submodels = TRUE,
        model_ids      = c(4, 5, 8, 9, 10), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem_2 %>% extract_nested_modeltime_table()

### Add more ensembles, as per the previous 2 but with "best" models adjusted to account for each series

#models 3, 4, 5, 6, 7, 13, 14, 15 for ts 2
ensem_3 <- ensem_2 %>%
    ensemble_nested_average(
        type           = "mean",
        keep_submodels = TRUE,
        model_ids      = c(3, 4, 5, 6, 7, 13, 14, 15), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem_3 %>% extract_nested_modeltime_table()

ensem_4 <- ensem_3 %>%
    ensemble_nested_average(
        type           = "median",
        keep_submodels = TRUE,
        model_ids      = c(3, 4, 5, 6, 7, 13, 14, 15), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem_4 %>% extract_nested_modeltime_table()

#### all models except 11 and 12 for ts 3
ensem_5 <- ensem_4 %>%
    ensemble_nested_average(
        type           = "mean",
        keep_submodels = TRUE,
        model_ids      = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem_5 %>% extract_nested_modeltime_table()

ensem_6 <- ensem_5 %>%
    ensemble_nested_average(
        type           = "median",
        keep_submodels = TRUE,
        model_ids      = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem_6 %>% extract_nested_modeltime_table()


#### models 1,2,3,7,8 for ts 4
ensem_7 <- ensem_6 %>%
    ensemble_nested_average(
        type           = "mean",
        keep_submodels = TRUE,
        model_ids      = c(1, 2, 3, 4, 9, 10), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem_7 %>% extract_nested_modeltime_table()

ensem_8 <- ensem_7 %>%
    ensemble_nested_average(
        type           = "median",
        keep_submodels = TRUE,
        model_ids      = c(1, 2, 3, 4, 9, 10), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )


#### models 1, 9, 10, 11 for ts 5
ensem_9 <- ensem_8 %>%
    ensemble_nested_average(
        type           = "mean",
        keep_submodels = TRUE,
        model_ids      = c(1, 9, 10, 11), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem_9 %>% extract_nested_modeltime_table()

ensem_10 <- ensem_9 %>%
    ensemble_nested_average(
        type           = "median",
        keep_submodels = TRUE,
        model_ids      = c(1, 9, 10, 11), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )


ensem_10 %>% extract_nested_modeltime_table()
 
#### models (2, 3, 6, 7, 9, 10, 13, 14, 15) for ts 6
ensem_11 <- ensem_10 %>%
    ensemble_nested_average(
        type           = "mean",
        keep_submodels = TRUE,
        model_ids      = c(2, 3, 6, 7, 9, 10, 13, 14, 15), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem_11 %>% extract_nested_modeltime_table()

ensem_12 <- ensem_11 %>%
    ensemble_nested_average(
        type           = "median",
        keep_submodels = TRUE,
        model_ids      = c(2, 3, 6, 7, 9, 10, 13, 14, 15), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem_12 %>% extract_nested_modeltime_table()

#### models 6, 7, 13, 14, 15 for ts 7

ensem_13 <- ensem_12 %>%
    ensemble_nested_average(
        type           = "mean",
        keep_submodels = TRUE,
        model_ids      = c(6, 7, 13, 14, 15), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem_13 %>% extract_nested_modeltime_table()

ensem_14 <- ensem_13 %>%
    ensemble_nested_average(
        type           = "median",
        keep_submodels = TRUE,
        model_ids      = c(6, 7, 13, 14, 15), # select the "best" models
        control        = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    )

ensem_14 %>% extract_nested_modeltime_table()

# weighted ensembles: timeseries 1

#loadings <- c(1, 1.01, 1.09 , 1.36, 1.15, 1.12)
model_ids <- c(4, 5, 8, 9, 10)

ensem_15 <- ensem_14 %>%
    ensemble_nested_weighted(
        loadings = c(4, 1, 3, 1, 1), # determine from mae of the chose models
        scale_loadings = TRUE,
        metric = "rmse",
        keep_submodels = TRUE,
        model_ids = model_ids,
        control = control_nested_fit(allow_par = TRUE, verbose = TRUE)
    )

# weighted ensembles: timeseries 2

#loadings <- c(1, 1.01, 1.09 , 1.36, 1.15, 1.12)
model_ids <- c(3, 4, 5, 6, 7, 13, 14, 15)

ensem_16 <- ensem_15 %>%
    ensemble_nested_weighted(
        loadings = c(1, 2, 1, 1, 3, 3, 3, 3), # determine from mae of the chose models
        scale_loadings = TRUE,
        metric = "rmse",
        keep_submodels = TRUE,
        model_ids = model_ids,
        control = control_nested_fit(allow_par = TRUE, verbose = TRUE)
    )

# weighted ensembles: timeseries 3

#loadings <- c(1, 1.01, 1.09 , 1.36, 1.15, 1.12)
model_ids <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15)

ensem_17 <- ensem_16 %>%
    ensemble_nested_weighted(
        loadings = c(2, 2, 2, 2, 2, 1, 3, 1, 1, 1, 3, 3, 3), # determine from mae of the chose models
        scale_loadings = TRUE,
        metric = "rmse",
        keep_submodels = TRUE,
        model_ids = model_ids,
        control = control_nested_fit(allow_par = TRUE, verbose = TRUE)
    )

# weighted ensembles: timeseries 4

#loadings <- c(1, 1.01, 1.09 , 1.36, 1.15, 1.12)
model_ids <- c(1, 2, 3, 4, 9, 10)

ensem_18 <- ensem_17 %>%
    ensemble_nested_weighted(
        loadings = c(2, 2, 2, 2, 1, 1), # determine from mae of the chose models
        scale_loadings = TRUE,
        metric = "rmse",
        keep_submodels = TRUE,
        model_ids = model_ids,
        control = control_nested_fit(allow_par = TRUE, verbose = TRUE)
    )

# weighted ensembles: timeseries 5

#loadings <- c(1, 1.01, 1.09 , 1.36, 1.15, 1.12)
model_ids <- c(1, 9, 10, 11)

ensem_19 <- ensem_18 %>%
    ensemble_nested_weighted(
        loadings = c(1, 3, 3, 2), # determine from mae of the chose models
        scale_loadings = TRUE,
        metric = "rmse",
        keep_submodels = TRUE,
        model_ids = model_ids,
        control = control_nested_fit(allow_par = TRUE, verbose = TRUE)
    )

# weighted ensembles: timeseries 6

#loadings <- c(1, 1.01, 1.09 , 1.36, 1.15, 1.12)
model_ids <- c(2, 3, 6, 7, 9, 10, 13, 14, 15)

ensem_20 <- ensem_19 %>%
    ensemble_nested_weighted(
        loadings = c(1, 1, 2, 1, 1, 3, 1, 1, 1), # determine from mae of the chose models
        scale_loadings = TRUE,
        metric = "rmse",
        keep_submodels = TRUE,
        model_ids = model_ids,
        control = control_nested_fit(allow_par = TRUE, verbose = TRUE)
    )

# weighted ensembles: timeseries 7

#loadings <- c(1, 1.01, 1.09 , 1.36, 1.15, 1.12)
model_ids <- c(4, 5, 96, 7, 13, 14, 15)

ensem_21 <- ensem_20 %>%
    ensemble_nested_weighted(
        loadings = c(1, 2, 1, 1, 1), # determine from mae of the chose models
        scale_loadings = TRUE,
        metric = "rmse",
        keep_submodels = TRUE,
        model_ids = model_ids,
        control = control_nested_fit(allow_par = TRUE, verbose = TRUE)
    )

## Review  Accuracy ----
#models only
nested_modeltime_tbl %>%
    extract_nested_test_accuracy() %>%
    table_modeltime_accuracy()

#ensemble accuracy
ensem_21 %>%
    extract_nested_test_accuracy() %>%
    table_modeltime_accuracy()

## Visualise Test Forecast ----
#models only
nested_modeltime_tbl %>%
    extract_nested_test_forecast() %>%
    #filter(item_id == "FOODS_3_090") %>%  # focus in this because it had the worst rmse
    group_by(id) %>%
    plot_modeltime_forecast(.facet_ncol = 3)

#ensemble included
ensem_21 %>%
    extract_nested_test_forecast() %>%
    #filter(item_id == "FOODS_3_090") %>%  # focus in this because it had the worst rmse
    group_by(id) %>%
    plot_modeltime_forecast(.facet_ncol = 3)


## Select Best ----
nested_best_tbl <- ensem_21 %>% #nested_modeltime_tbl
    modeltime_nested_select_best(metric = "rmse") # select any from mae, mape, mase, smape, rmse

## Visualise Best Models ----
nested_best_tbl %>%
    extract_nested_test_forecast() %>%
    #filter(as.numeric(item_id) %in% 1:12) %>% #for demo, just for first time series
    group_by(id) %>%
    plot_modeltime_forecast(.facet_ncol = 3)

## Refit ----
# long running script

nested_best_refit_tbl <- nested_best_tbl %>%
    modeltime_nested_refit(
        control = control_refit(
            verbose = TRUE,
            allow_par = TRUE
        )
    )

# nested_best_refit_tbl %>% write_rds("nested_best_refit_tbl.rds")
# nested_best_refit_tbl <- read_rds("nested_best_refit_tbl.rds")

## review any errors ---
nested_best_refit_tbl %>% extract_nested_error_report() # if there are errors


## Visualise Future Forecast ----
nested_best_refit_tbl %>%
    extract_nested_future_forecast() %>%
    group_by(id) %>%
    plot_modeltime_forecast(.facet_ncol = 3)

nested_best_refit_tbl %>%
    extract_nested_future_forecast() %>%
    #group_by(id) %>%
    filter(id == "1_93") %>%   
    plot_modeltime_forecast(.facet_ncol = 1)

extract_nested_future_forecast((nested_best_refit_tbl[[5]][[6]]))#[[5]][[1]])

parallel_stop()



