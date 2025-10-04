library(readr)
library(dplyr)
library(tidyr)
library(forcats)
library(ggplot2)
library(stringr)

read_approximators_benchmark <- function(
    benchmark_name
) {
    result_data <<- read_csv(
        file = paste0('./results/approximators_', benchmark_name, '.csv'),
        col_types = cols(
            Game = col_factor(),
            k = col_integer(),
            Approximator = col_factor(),
            Variant = col_factor(),
            Budget = col_integer(),
            MSE_mean = col_double(),
            MSE_std = col_double(),
            MSE_min = col_double(),
            MSE_max = col_double(),
            MSE_se = col_double(),
            Prec10_mean = col_double(),
            Prec10_std = col_double(),
            Prec10_min = col_double(),
            Prec10_max = col_double(),
            Prec10_se = col_double(),
            Runtime_mean = col_double(),
            Runtime_std = col_double(),
            Runtime_min = col_double(),
            Runtime_max = col_double(),
            Runtime_se = col_double()
        )
    )
    last_benchmark_name <<- benchmark_name
    
    return(result_data)
}

plot_approximators_mse <- function(
    benchmark_name,
    ncol = 4,
    facet_label_wrap_width = 30,
    logarithmic = TRUE
) {
  plot_data <<- read_approximators_benchmark(benchmark_name) |>
    filter(Budget >= 50) |>
    filter(is.na(Variant) | Variant == "inverse_variance_weighting") |>
    mutate(
      facetvar = paste0("k=", k)
    )
  
  plot <- ggplot(data = plot_data, mapping = aes(x = Budget, y = MSE_mean, group = Approximator)) +
    geom_ribbon(aes(ymin = pmax(MSE_mean - MSE_se, 0), ymax = MSE_mean + MSE_se, fill = Approximator), alpha = 0.1, color = NA) +
    geom_line(aes(color = Approximator)) +
    facet_wrap(vars(facetvar), scales = "free", ncol = ncol, labeller = label_wrap_gen(facet_label_wrap_width)) +
    scale_y_continuous(
      labels = scales::label_number_auto(),
      trans = ifelse(logarithmic, 'log10', 'identity')
    ) +
    theme(strip.clip = "off", strip.background = element_blank()) +
    theme(legend.position = "bottom", legend.direction = "vertical") +
    expand_limits(x = 0) +
    list()
  
  print(plot)
  
  basename <- paste0("./images/approximators_", benchmark_name, "_mse")
  ggsave(
    filename = paste0(basename, ".png"),
    plot = plot,
    width = 24,
    height = 16,
    units = "cm",
    dpi = 300
  )
  print(paste0("Saved plot to: ", basename, ".png"))
}

plot_approximators_prec10 <- function(
    benchmark_name,
    ncol = 4,
    facet_label_wrap_width = 30,
    logarithmic = FALSE
) {
  plot_data <<- read_approximators_benchmark(benchmark_name) |>
    filter(is.na(Variant) | Variant == "inverse_variance_weighting") |>
    mutate(
      facetvar = paste0("k=", k)
    )
  
  plot <- ggplot(data = plot_data, mapping = aes(x = Budget, y = Prec10_mean, group = Approximator)) +
    geom_ribbon(aes(ymin = pmax(Prec10_mean - Prec10_se, 0), ymax = Prec10_mean + Prec10_se, fill = Approximator), alpha = 0.1, color = NA) +
    geom_line(aes(color = Approximator)) +
    facet_wrap(vars(facetvar), scales = "free", ncol = ncol, labeller = label_wrap_gen(facet_label_wrap_width)) +
    scale_y_continuous(
      labels = scales::label_number_auto(),
      trans = ifelse(logarithmic, 'log10', 'identity')
    ) +
    theme(strip.clip = "off", strip.background = element_blank()) +
    theme(legend.position = "bottom", legend.direction = "vertical") +
    expand_limits(x = 0) +
    list()
  
  print(plot)
  
  basename <- paste0("./images/approximators_", benchmark_name, "_prec10")
  ggsave(
    filename = paste0(basename, ".png"),
    plot = plot,
    width = 24,
    height = 16,
    units = "cm",
    dpi = 300
  )
  print(paste0("Saved plot to: ", basename, ".png"))
}

plot_approximators_mse("localexplanation_adultcensus")
plot_approximators_prec10("localexplanation_adultcensus")

plot_approximators_mse("globalexplanation_adultcensus")
plot_approximators_prec10("globalexplanation_adultcensus")

plot_approximators_mse("imageclassifier_n14")
plot_approximators_prec10("imageclassifier_n14")

plot_approximators_mse("imageclassifier_n16")
plot_approximators_prec10("imageclassifier_n16")

plot_approximators_mse("unsupervisedfeatureimportance_adultcensus")
plot_approximators_prec10("unsupervisedfeatureimportance_adultcensus")

plot_approximators_mse("datasetvaluation_californiahousing")
plot_approximators_prec10("datasetvaluation_californiahousing")


