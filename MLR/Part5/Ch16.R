################################################################
# Description: Building your first k-means model
# Author(s): Stephen CUI
# LastEditor(s): Stephen CUI
# CreatedTime: 2023-06-29 12:57:46
################################################################
library(mlr)
library(tidyverse)
data(GvHD, package = "mclust")
gvhdTib <- as_tibble(GvHD.control)
gvhdScaled <- gvhdTib %>% scale()
library(GGally)
ggpairs(GvHD.control,
    upper = list(continuous = "density"),
    lower = list(continuous = wrap("points", size = .5)),
    diag = list(continuous = "densityDiag")
) + theme_bw()
