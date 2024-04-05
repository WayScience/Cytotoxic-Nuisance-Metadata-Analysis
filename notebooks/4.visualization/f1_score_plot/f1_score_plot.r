library(ggplot2)

# loading in f1_score results
f1_score_path <- file.path("../../../results/2.modeling/all_f1_scores.csv.gz")
f1_df <- read.csv(f1_score_path, sep = ",")

# replacing True and False to shuffled to non shuffled
f1_df$shuffled <- ifelse(f1_df$shuffled == "False", "Not Shuffled",
                                      ifelse(f1_df$shuffled == "True", "Shuffled", f1_df$shuffled))

# display dimensions
print(dim(f1_df))
head(f1_df)

# image size
img_height <- 10
img_width <- 10

options(repr.plot.width = img_width, repr.plot.height = img_height)
# creating a bar plot with a facet grid dictated weather the model has been shuffled or not
# along the y, facet grid was generated based on the dataset type
legend_order <- c("test", "train", "plate_holdout", "treatment_holdout", "well_holdout")

f1_plot <- ggplot(f1_df, aes(x = shuffled, y = f1_score, fill = factor(dataset_type, levels = legend_order))) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_bw() +
    ylim(0, 1) +
    facet_wrap(~injury_type) +
    labs(y = "F1 Score", x = "Data Split", fill = "Datasets")

ggsave(filename = "f1_score_barplots.png", height = img_height, width = img_width)
f1_plot
