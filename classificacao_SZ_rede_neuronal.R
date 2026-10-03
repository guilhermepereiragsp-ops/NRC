# =========================================================
# 1. Instalar e carregar pacotes
# =========================================================
if(!require(readxl)) install.packages("readxl")
if(!require(nnet)) install.packages("nnet")
if(!require(NeuralNetTools)) install.packages("NeuralNetTools")
if(!require(ggplot2)) install.packages("ggplot2")

library(readxl)
library(nnet)
library(NeuralNetTools)
library(ggplot2)

# =========================================================
# 2. Ler dados do Excel (folha específica)
# =========================================================
file_path <- "C:/Users/gui_m/Desktop/NRC/Data_Merged_Demo.xlsx"
df <- read_excel(file_path, sheet = "FinalData")
df <- as.data.frame(df)

# =========================================================
# 3. Converter target e variáveis demográficas
# =========================================================
# Target
df$group <- factor(df$group, levels = c(0,1))

# Gender → numérico
df$gender <- ifelse(df$gender %in% c("M","Male","male"), 0,
                    ifelse(df$gender %in% c("F","Female","female"), 1, NA))

# =========================================================
# 4. Selecionar features
# =========================================================
# Todas as features testadas
# features <- c(
#   "gender", "age", "education", "ΔFz_N100", "ΔFCz_N100", "ΔCz_N100", "ΔFC3_N100",
#   "ΔFC4_N100", "ΔC3_N100", "ΔC4_N100", "ΔCP3_N100", "ΔCP4_N100"
# )

# Features do modelo final
features <- c(
  "ΔFz_N100", "ΔFCz_N100", "ΔCz_N100", "ΔFC3_N100",
  "ΔFC4_N100", "ΔC3_N100", "ΔC4_N100", "ΔCP3_N100", "ΔCP4_N100"
)

df_model <- df[, c(features, "group")]

# =========================================================
# 5. Remover NA (OBRIGATÓRIO)
# =========================================================
df_model <- na.omit(df_model)

cat("Número de amostras após limpeza:", nrow(df_model), "\n")

# =========================================================
# 6. Normalização (z-score)
# =========================================================
df_model[features] <- scale(df_model[features])

# Verificação final
stopifnot(!any(is.na(df_model)))
stopifnot(!any(is.infinite(as.matrix(df_model[features]))))

# =========================================================
# 7. Divisão estratificada (80/20) + (60/20)
# =========================================================
set.seed(123)

train_val <- data.frame()
test_data <- data.frame()

for(g in levels(df_model$group)){
  gdata <- df_model[df_model$group == g, ]
  n <- nrow(gdata)
  idx <- sample(1:n, n)
  cut <- round(0.8 * n)
  
  train_val <- rbind(train_val, gdata[idx[1:cut], ])
  test_data  <- rbind(test_data,  gdata[idx[(cut+1):n], ])
}

train_data <- data.frame()
val_data <- data.frame()

for(g in levels(train_val$group)){
  gdata <- train_val[train_val$group == g, ]
  n <- nrow(gdata)
  idx <- sample(1:n, n)
  cut <- round(0.75 * n)
  
  train_data <- rbind(train_data, gdata[idx[1:cut], ])
  val_data   <- rbind(val_data,  gdata[idx[(cut+1):n], ])
}

# =========================================================
# 8. Treino do modelo (backpropagation - nnet)
# =========================================================
neurons <- c(3, 5, 7, 10)
best_acc <- 0
best_nn <- NULL
best_size <- NULL

for(size in neurons){
  set.seed(123)
  nn <- nnet(
    x = train_data[, features],
    y = class.ind(train_data$group),
    size = size,
    maxit = 300,
    decay = 1e-3,
    trace = FALSE
  )
  
  val_prob <- predict(nn, val_data[, features])[,2]
  val_pred <- ifelse(val_prob > 0.5, 1, 0)
  
  acc <- mean(val_pred == as.numeric(as.character(val_data$group)))
  
  if(acc > best_acc){
    best_acc <- acc
    best_nn <- nn
    best_size <- size
  }
}

# =========================================================
# 9. Plot da rede neuronal
# =========================================================
plotnet(best_nn, alpha = 0.6, circle_col = "lightblue", show_weights = TRUE)

cat("Melhor nº de neurónios ocultos:", best_size, "\n")
cat("Acurácia validação:", round(best_acc, 4), "\n")

# =========================================================
# 10. Avaliação no conjunto de teste
# =========================================================
test_prob <- predict(best_nn, test_data[, features])[,2]
threshold <- 0.5
test_pred <- ifelse(test_prob > threshold, 1, 0)

test_acc <- mean(test_pred == as.numeric(as.character(test_data$group)))
cat("Acurácia TESTE FINAL:", round(test_acc, 4), "\n")

# Matriz de confusão
conf_matrix <- table(Predicted = test_pred,
                     Actual = as.numeric(as.character(test_data$group)))
print(conf_matrix)

# =========================================================
# 11. Plot do perceptron (threshold + concordância)
# =========================================================
plot_df <- data.frame(
  Actual = as.numeric(as.character(test_data$group)),
  Prob = test_prob,
  Pred = test_pred
)

plot_df$Match <- ifelse(plot_df$Actual == plot_df$Pred,
                        "Concordância", "Discordância")

ggplot(plot_df, aes(x = Actual, y = Prob, color = Match)) +
  geom_jitter(width = 0.2, height = 0, size = 2, alpha = 0.7) +
  geom_hline(yintercept = threshold, linetype = "dashed", color = "red") +
  labs(
    title = "Saída do Perceptron – Classificação SZ vs Controlo",
    x = "Classe Real (0=Controlo, 1=SZ)",
    y = "Probabilidade prevista (classe SZ)"
  ) +
  scale_color_manual(values = c("Concordância" = "blue",
                                "Discordância" = "orange")) +
  theme_minimal()

