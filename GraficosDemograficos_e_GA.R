# =============================
# 01_graficos.R
# Gráficos descritivos (demografia)
# =============================

if(!require(readxl)) install.packages("readxl")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(dplyr)) install.packages("dplyr")

library(readxl)
library(ggplot2)
library(dplyr)

# Caminho do ficheiro
file_path <- "C:/Users/gui_m/Desktop/GitHub/NRC/data/Data_Merged_Demo.xlsx"

# Ler sheet 1 (demografia)
demo <- read_excel(file_path, sheet = 1) |> as.data.frame()

# Garantir factores
demo$group <- factor(demo$group, levels = c(0, 1),
                     labels = c("Controlo", "Esquizofrenia"))
demo$gender <- factor(demo$gender)

# Número de sujeitos por grupo
ggplot(demo, aes(x = group, fill = group)) +
  geom_bar(width = 0.6) +
  labs(title = "Número de Sujeitos por Grupo",
       x = "Grupo",
       y = "Número de Sujeitos") +
  theme_minimal() +
  theme(legend.position = "none")

# Distribuição de género por grupo
ggplot(demo, aes(x = gender, fill = group)) +
  geom_bar(position = "dodge") +
  labs(title = "Distribuição de Género por Grupo",
       x = "Género",
       y = "Número de Sujeitos",
       fill = "Grupo") +
  theme_minimal()

# Idade por grupo
ggplot(demo, aes(x = group, y = age, fill = group)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.4) +
  labs(title = "Distribuição da Idade por Grupo",
       x = "Grupo",
       y = "Idade (anos)") +
  theme_minimal() +
  theme(legend.position = "none")

# Educação por grupo
ggplot(demo, aes(x = group, y = education, fill = group)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.4) +
  labs(title = "Anos de Educação por Grupo",
       x = "Grupo",
       y = "Anos de Educação") +
  theme_minimal() +
  theme(legend.position = "none")

# Idade vs Educação
ggplot(demo, aes(x = age, y = education, colour = group)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = "Idade vs Educação",
       x = "Idade",
       y = "Anos de Educação",
       colour = "Grupo") +
  theme_minimal()


# =============================
# 02_GA_deltas.R
# Algoritmo Genético (só deltas), validação por sujeito
# =============================

packs <- c("readxl","dplyr","GA","caret","nnet","pROC")
to_install <- packs[!packs %in% installed.packages()[, "Package"]]
if(length(to_install)) install.packages(to_install)

library(readxl)
library(dplyr)
library(GA)
library(caret)
library(nnet)
library(pROC)

set.seed(123)

# Ler dados (sheet 3)
file_path <- "C:/Users/gui_m/Desktop/GitHub/NRC/data/Data_Merged_Demo.xlsx"
df <- read_excel(file_path, sheet = 3) |> as.data.frame()

# Target
df$group <- factor(df$group, levels = c(0, 1))

# ID do sujeito
subject_id <- as.character(df$subject_0)
n_subj <- length(unique(subject_id))
cat("Número de sujeitos únicos:", n_subj, "\n")

k <- min(5, n_subj)
if(k < 2) stop("Menos de 2 sujeitos — verifica a coluna subject_0")

# Seleccionar todas as colunas Δ
delta_cols <- grep("^Δ", names(df), value = TRUE)
cat("Nº total de features Δ:", length(delta_cols), "\n")

X_raw <- df[, delta_cols, drop = FALSE]

# Garantir numérico
X_raw <- as.data.frame(lapply(X_raw, function(x) {
  if(is.character(x)) x <- gsub(",", ".", x)
  suppressWarnings(as.numeric(x))
}))

# Remover colunas near-zero e colunas quase vazias
nzv <- nearZeroVar(X_raw, saveMetrics = TRUE)
X_raw <- X_raw[, !nzv$nzv, drop = FALSE]
X_raw <- X_raw[, colSums(is.na(X_raw)) < nrow(X_raw), drop = FALSE]

cat("Features após limpeza:", ncol(X_raw), "\n")

y <- df$group

# Folds por sujeito
folds <- groupKFold(subject_id, k = k)

# Função de fitness (AUC média)
fitness_auc <- function(chromosome) {
  idx <- which(chromosome == 1)
  if(length(idx) < 2) return(0)

  feat_names <- colnames(X_raw)[idx]
  aucs <- c()

  for(i in seq_along(folds)) {
    test_idx  <- folds[[i]]
    train_idx <- setdiff(seq_len(nrow(X_raw)), test_idx)

    Xtr0 <- X_raw[train_idx, feat_names, drop = FALSE]
    Xte0 <- X_raw[test_idx,  feat_names, drop = FALSE]
    ytr  <- y[train_idx]
    yte  <- y[test_idx]

    # Sem leakage: pré-processamento só no treino
    pp <- preProcess(Xtr0, method = c("medianImpute","center","scale"))
    Xtr <- predict(pp, Xtr0)
    Xte <- predict(pp, Xte0)

    model <- try(
      nnet(
        x = as.matrix(Xtr),
        y = class.ind(ytr),
        size = 5,
        decay = 1e-3,
        maxit = 300,
        trace = FALSE
      ),
      silent = TRUE
    )
    if(inherits(model, "try-error")) return(0)

    probs <- predict(model, as.matrix(Xte), type = "raw")[, "1"]
    roc_obj <- try(roc(yte, probs, quiet = TRUE), silent = TRUE)
    if(inherits(roc_obj, "try-error")) return(0)

    aucs <- c(aucs, as.numeric(auc(roc_obj)))
  }

  mean_auc <- mean(aucs)
  penalty <- 0.001 * sum(chromosome)
  mean_auc - penalty
}

# GA
n_feat <- ncol(X_raw)
ga_res <- ga(
  type = "binary",
  fitness = fitness_auc,
  nBits = n_feat,
  popSize = 40,
  maxiter = 50,
  run = 15,
  pmutation = 0.1,
  elitism = 2,
  keepBest = TRUE
)

# Resultado
best <- ga_res@solution[1, ]
best_features <- colnames(X_raw)[which(best == 1)]

cat("\n=== MELHOR SOLUÇÃO (GA) ===\n")
cat("Fitness (AUC com penalização):", ga_res@fitnessValue, "\n")
cat("Nº features escolhidas:", length(best_features), "\n")
print(best_features)

cat("\nResumo do GA:\n")
print(summary(ga_res))


# =============================
# 03_GA_deltas_demografia_centrada.R
# Algoritmo Genético: deltas + demografia centrada por sujeito
# =============================

packs <- c("readxl","dplyr","GA","caret","nnet","pROC")
to_install <- packs[!packs %in% installed.packages()[, "Package"]]
if(length(to_install)) install.packages(to_install)

library(readxl)
library(dplyr)
library(GA)
library(caret)
library(nnet)
library(pROC)

set.seed(123)

# Ler dados (sheet 3)
file_path <- "C:/Users/maria/Downloads/Data_Merged_Demo.xlsx"
df <- read_excel(file_path, sheet = 3) |> as.data.frame()

# Target
df$group <- factor(df$group, levels = c(0, 1))

# ID do sujeito
subject_id <- as.character(df$subject_0)
n_subj <- length(unique(subject_id))
cat("Número de sujeitos únicos:", n_subj, "\n")

k <- min(5, n_subj)
if(k < 2) stop("Menos de 2 sujeitos — verifica a coluna subject_0")

# Centrar demografia por sujeito (remove identificação directa)
df <- df %>%
  group_by(subject_0) %>%
  mutate(
    age_c = as.numeric(age) - mean(as.numeric(age), na.rm = TRUE),
    education_c = as.numeric(education) - mean(as.numeric(education), na.rm = TRUE)
  ) %>%
  ungroup()

# Features: demografia centrada + deltas
delta_cols <- grep("^Δ", names(df), value = TRUE)
feature_cols <- c("age_c","education_c", delta_cols)

cat("Nº colunas Δ:", length(delta_cols), "\n")
cat("Nº total features candidatas:", length(feature_cols), "\n")

X_raw <- df[, feature_cols, drop = FALSE]

# Garantir numérico
X_raw <- as.data.frame(lapply(X_raw, function(x) {
  if(is.character(x)) x <- gsub(",", ".", x)
  suppressWarnings(as.numeric(x))
}))

# Remover colunas near-zero e colunas quase vazias
nzv <- nearZeroVar(X_raw, saveMetrics = TRUE)
X_raw <- X_raw[, !nzv$nzv, drop = FALSE]
X_raw <- X_raw[, colSums(is.na(X_raw)) < nrow(X_raw), drop = FALSE]

cat("Features após limpeza:", ncol(X_raw), "\n")

y <- df$group

# Folds por sujeito
folds <- groupKFold(subject_id, k = k)

# Função de fitness (AUC média)
fitness_auc <- function(chromosome) {
  idx <- which(chromosome == 1)
  if(length(idx) < 2) return(0)

  feat_names <- colnames(X_raw)[idx]
  aucs <- c()

  for(i in seq_along(folds)) {
    test_idx  <- folds[[i]]
    train_idx <- setdiff(seq_len(nrow(X_raw)), test_idx)

    Xtr0 <- X_raw[train_idx, feat_names, drop = FALSE]
    Xte0 <- X_raw[test_idx,  feat_names, drop = FALSE]
    ytr  <- y[train_idx]
    yte  <- y[test_idx]

    # Sem leakage: pré-processamento só no treino
    pp <- preProcess(Xtr0, method = c("medianImpute","center","scale"))
    Xtr <- predict(pp, Xtr0)
    Xte <- predict(pp, Xte0)

    model <- try(
      nnet(
        x = as.matrix(Xtr),
        y = class.ind(ytr),
        size = 5,
        decay = 1e-3,
        maxit = 300,
        trace = FALSE
      ),
      silent = TRUE
    )
    if(inherits(model, "try-error")) return(0)

    probs <- predict(model, as.matrix(Xte), type = "raw")[, "1"]
    roc_obj <- try(roc(yte, probs, quiet = TRUE), silent = TRUE)
    if(inherits(roc_obj, "try-error")) return(0)

    aucs <- c(aucs, as.numeric(auc(roc_obj)))
  }

  mean_auc <- mean(aucs)
  penalty <- 0.001 * sum(chromosome)
  mean_auc - penalty
}

# GA
n_feat <- ncol(X_raw)
ga_res <- ga(
  type = "binary",
  fitness = fitness_auc,
  nBits = n_feat,
  popSize = 40,
  maxiter = 50,
  run = 15,
  pmutation = 0.1,
  elitism = 2,
  keepBest = TRUE
)

# Resultado
best <- ga_res@solution[1, ]
best_features <- colnames(X_raw)[which(best == 1)]

cat("\n=== MELHOR SOLUÇÃO (GA) ===\n")
cat("Fitness (AUC com penalização):", ga_res@fitnessValue, "\n")
cat("Nº features escolhidas:", length(best_features), "\n")
print(best_features)

cat("\nResumo do GA:\n")
print(summary(ga_res))

