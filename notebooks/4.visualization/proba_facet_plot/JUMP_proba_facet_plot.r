# import libraries
library(ggplot2)
suppressPackageStartupMessages(suppressWarnings(library(RColorBrewer))) # color palettes
suppressPackageStartupMessages(suppressWarnings(library(ggridges))) # ridgeline plots

# adding the file path probability file
proba_path <- file.path("../../../results/3.jump_analysis/JUMP_injury_proba.csv.gz")

# loading in probability file path
proba_df <- read.table(proba_path, head = TRUE, sep=",")

# replacing True and False to shuffled to non shuffled
proba_df$shuffled_model <- ifelse(proba_df$shuffled_model == "False", "Not Shuffled",
                                      ifelse(proba_df$shuffled_model == "True", "Shuffled", proba_df$shuffled_model))
# displaying
print(dim(proba_df))
head(proba_df)

# showing the unique injuries
unique(proba_df$injury_type)

# image size
img_height <- 10
img_width <- 15

options(repr.plot.width = img_width, repr.plot.height = img_height)
ridge_plot <- (
    ggplot(proba_df, aes(x = proba, y = pred_injury, fill = shuffled_model)) +
    geom_density_ridges() +
    # facet_grid(shuffled_model~injury_type, scales = "free_y") +
    facet_wrap(~injury_type) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "black") +
    scale_x_continuous(breaks = seq(0, 1, 0.5)) +
    labs(title = "JUMP Injury Prediction Probability", y = "Injury Types", x = "Probability", fill = "Model Type") +
    theme_bw() +

    # no legend
    theme(plot.title = element_text(size = 20, hjust = 0.5)) +

    scale_fill_manual(values = c(

        "Shuffled" = brewer.pal(2, "Dark2")[2],
        "Not Shuffled" = "#1E93FC"
        # "Not Shuffled" = brewer.pal(2, "Dark2")[1]
    )) +

    # remove x axis label
    theme(axis.title.x = element_blank())
    )
ridge_plot

ggsave(filename = "JUMP_cell_injury_facet_proba.png", height = img_height, width = img_width, dpi=600)
