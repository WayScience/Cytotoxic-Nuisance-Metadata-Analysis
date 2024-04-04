# import libraries
library(ggplot2)
suppressPackageStartupMessages(suppressWarnings(library(ggridges))) # ridgeline plots

# adding the file path probability file
proba_path <- file.path("../../../results/3.jump_analysis/JUMP_injury_proba.csv.gz")

# loading in probability file path
proba_df <- read.table(proba_path, head = TRUE, sep=",")

# displaying
print(dim(proba_df))
head(proba_df)

# showing the unique injuries
unique(proba_df$injury_type)

# image size

img_height <- 10
img_width <- 15

options(repr.plot.width = width, repr.plot.height = height)
ridge_plot <- (
    ggplot(proba_df, aes(x = proba, y = pred_injury)) +
    geom_density_ridges() +
    facet_grid(shuffled_model~injury_type, scales = "free_y") +
    geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
    scale_x_continuous(breaks = seq(0, 1, 0.5)) +
    labs(title = "JUMP Injury Prediction Probability", y = "Injury Types", x = "Probability") +
    # + labs()
    theme_bw() +

    # no legend
    # + theme(legend.position = "none")
    theme(plot.title = element_text(size = 20, hjust = 0.5)) +

    # remove x axis label
    theme(axis.title.x = element_blank())
    )
ridge_plot

ggsave(filename = "JUMP_cell_injury_facet_proba.png", height = img_height, width = img_width)
