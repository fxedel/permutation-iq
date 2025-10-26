library(readr)
library(dplyr)
library(tidyr)
library(forcats)
library(ggplot2)
library(stringr)

read_permutationiq_variants_benchmark <- function(
    benchmark_name
) {
  result_data <<- read_csv(
    file = paste0('./results/permutationiq_variants_', benchmark_name, '.csv'),
    col_types = cols(
      Game = col_factor(),
      k = col_integer(),
      Group = col_factor(),
      Approximator = col_factor(),
      Variant = col_factor(),
      Budget = col_integer(),
      ErrorSquared_mean = col_double(),
      ErrorSquared_std = col_double(),
      ErrorSquared_min = col_double(),
      ErrorSquared_max = col_double(),
      ErrorSquared_se = col_double(),
      Error_mean = col_double(),
      Error_std = col_double(),
      Error_min = col_double(),
      Error_max = col_double(),
      Error_se = col_double()
    )
  )
  last_benchmark_name <<- benchmark_name
  
  return(result_data)
}

plot_permutationiq_variants_mse <- function(
    benchmark_name,
    groups = c(
      "(0, 1)", "(2, 3)", "(4, 5)", "(6, 7)", "(0, 2)", "(1, 3)", "(4, 6)", "(5, 7)",
      "(0, 1, 2)", "(3, 4, 5)", "(6, 7, 8)", "(9, 10, 11)"
    ),
    ncol = 4,
    facet_label_wrap_width = 30,
    logarithmic = TRUE
) {
  monochrome_approximator_variant_prefix <- "player_"
  monochrome_approximator_variant_label <- "Player-specific\napproximator variants"
  colored_approximator_variant_label <- "Aggregated\napproximator variants"

  plot_data <<- read_permutationiq_variants_benchmark(benchmark_name) |>
    filter(Budget >= 50) |>
    filter(Group %in% groups) |>
    mutate(
      facetvar = paste0(Game, ", K=", Group)
    )
  
  data_colored <- plot_data |> filter(!(startsWith(as.character(Variant), monochrome_approximator_variant_prefix)))
  data_monochrome <- plot_data |> filter(startsWith(as.character(Variant), monochrome_approximator_variant_prefix))

  plot <- ggplot(data = plot_data, mapping = aes(x = Budget, y = ErrorSquared_mean, group = Variant)) +
    geom_ribbon(data = data_colored, aes(ymin = pmax(ErrorSquared_mean - ErrorSquared_se, 0), ymax = ErrorSquared_mean + ErrorSquared_se, fill = Variant), alpha = 0.1, color = NA) +
    geom_line(data = data_colored, aes(color = Variant)) +
    geom_line(data = data_monochrome, aes(lty = Variant), alpha = 0.5) +
    facet_wrap(vars(facetvar), scales = "free", ncol = ncol, labeller = label_wrap_gen(facet_label_wrap_width)) +
    scale_y_continuous(
      labels = scales::label_number_auto(),
      trans = ifelse(logarithmic, 'log10', 'identity')
    ) +
    theme(strip.clip = "off", strip.background = element_blank()) +
    theme(legend.position = "right", legend.direction = "vertical") +
    labs(color = colored_approximator_variant_label, fill = colored_approximator_variant_label, lty = monochrome_approximator_variant_label) +
    expand_limits(x = 0) +
    list()
  
  print(plot)
  
  basename <- paste0("./images/permutationiq_variants_", benchmark_name, "_mse")
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

plot_runtime_soum_varying_k <- function(
    benchmark_name,
    ncol = 4,
    facet_label_wrap_width = 30,
    logarithmic = TRUE
) {
  result_data <<- read_csv(
    file = paste0('./results/runtime_soum_varying_k.csv'),
    col_types = cols(
      Game = col_factor(),
      n = col_integer(),
      k = col_integer(),
      Approximator = col_factor(),
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
      Runtime_se = col_double(),
      Iterations = col_integer()
    )
  )

  plot_data <<- result_data |>
    mutate(
      facetvar = paste0("Game=", Game, ", n=", n, ", Budget=", Budget)
    )
  
  plot <- ggplot(data = plot_data, mapping = aes(x = k, y = Runtime_mean, group = Approximator)) +
    geom_ribbon(aes(ymin = pmax(Runtime_mean - Runtime_se, 0), ymax = Runtime_mean + Runtime_se, fill = Approximator), alpha = 0.1, color = NA) +
    geom_line(aes(color = Approximator)) +
    facet_wrap(vars(facetvar), scales = "free", ncol = ncol, labeller = label_wrap_gen(facet_label_wrap_width)) +
    scale_y_continuous(
      labels = scales::label_number_auto(),
      trans = ifelse(logarithmic, 'log10', 'identity')
    ) +
    theme(strip.clip = "off", strip.background = element_blank()) +
    theme(legend.position = "bottom", legend.direction = "vertical") +
    list()
  
  print(plot)
  
  basename <- paste0("./images/runtime_soum_varying_k")
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

plot_runtime_soum_varying_n <- function(
    benchmark_name,
    ncol = 4,
    facet_label_wrap_width = 30,
    logarithmic = TRUE
) {
  result_data <<- read_csv(
    file = paste0('./results/runtime_soum_varying_n.csv'),
    col_types = cols(
      Game = col_factor(),
      n = col_integer(),
      k = col_integer(),
      Approximator = col_factor(),
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
      Runtime_se = col_double(),
      Iterations = col_integer()
    )
  )

  plot_data <<- result_data |>
    mutate(
      facetvar = paste0("Game=", Game, ", k=", k, ", Budget=", Budget)
    )
  
  plot <- ggplot(data = plot_data, mapping = aes(x = n, y = Runtime_mean, group = Approximator)) +
    geom_ribbon(aes(ymin = pmax(Runtime_mean - Runtime_se, 0), ymax = Runtime_mean + Runtime_se, fill = Approximator), alpha = 0.1, color = NA) +
    geom_line(aes(color = Approximator)) +
    facet_wrap(vars(facetvar), scales = "free", ncol = ncol, labeller = label_wrap_gen(facet_label_wrap_width)) +
    scale_y_continuous(
      labels = scales::label_number_auto(),
      trans = ifelse(logarithmic, 'log10', 'identity')
    ) +
    theme(strip.clip = "off", strip.background = element_blank()) +
    theme(legend.position = "bottom", legend.direction = "vertical") +
    list()
  
  print(plot)
  
  basename <- paste0("./images/runtime_soum_varying_n")
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

plot_permutationiq_variants_mse("localexplanation_adultcensus")
plot_permutationiq_variants_mse("globalexplanation_adultcensus")
plot_permutationiq_variants_mse("soum")


# plot_approximators_mse("localexplanation_adultcensus")
# plot_approximators_prec10("localexplanation_adultcensus")

# plot_approximators_mse("globalexplanation_adultcensus")
# plot_approximators_prec10("globalexplanation_adultcensus")

# plot_approximators_mse("imageclassifier_n14")
# plot_approximators_prec10("imageclassifier_n14")

# # plot_approximators_mse("imageclassifier_n16")
# # plot_approximators_prec10("imageclassifier_n16")

# plot_approximators_mse("unsupervisedfeatureimportance_adultcensus")
# plot_approximators_prec10("unsupervisedfeatureimportance_adultcensus")

# plot_approximators_mse("datasetvaluation_californiahousing")
# plot_approximators_prec10("datasetvaluation_californiahousing")

# plot_runtime_soum_varying_k()
# plot_runtime_soum_varying_n()

