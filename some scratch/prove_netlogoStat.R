
# ora ricomincio con nuovo DF#####

library(readr)
# aggiunti simboli commento alle righe che non dovevo prendere del file "# "


WSP_DaFr <- read_csv("C:/Users/pc/Downloads/run_tre.csv", 
                       col_names = FALSE, 
                       comment = "#")
# View(WSP_DaFr) # se il df è grosso se pianta tutto
View(WSP_DaFr[1:100,1:100])

WSP_DaFr <- as.data.frame(WSP_DaFr)

WSP_DaFr <- WSP_DaFr[-(2:13),]
nomi <- WSP_DaFr[2]
WSP_DaFr <- WSP_DaFr[-2,]
primestat_defaultNetLo <- WSP_DaFr[2:5,]
WSP_DaFr <- WSP_DaFr[-(1:7),]
WSP_DaFr <- WSP_DaFr[,-1]
WSP_DaFr_bku <- WSP_DaFr
# dim: 301 14000
WSP_DaFr <- array(as.matrix(WSP_DaFr), c(301,14,1000))
# > dim(WSP_DaFr)
#[1]  301   14 1000
###################
# for (c in 1:length(WSP_DaFr[2,])) {
#   if (is.na(WSP_DaFr[2,c])) {
#     WSP_DaFr[2,c] <- pivot
#   }
#   else {
#     pivot <- WSP_DaFr[2,c]
#   }
#   print(WSP_DaFr[2,c])
# }
# 
# for (c in 1:length(WSP_DaFr[3,])) {
#   if (is.na(WSP_DaFr[3,c])) {
#     WSP_DaFr[3,c] <- pivot
#   }
#   else {
#     pivot <- WSP_DaFr[3,c]
#   }
#   print(WSP_DaFr[3,c])
# }
# 
# WSP_DaFr[6:length(WSP_DaFr[,1]), 1] <- as.character(0:(length(WSP_DaFr[,1]) - 6))
# 
# 
# nomi_righe <- WSP_DaFr[,1]
# 
# WSP_DaFr <- WSP_DaFr[,-1]
# 
# WSP_DaFr[4,] <- as.character(rep(c(1,2,3),9))
# 
# #controllo
# sum(is.na(WSP_DaFr))

write.csv(WSP_DaFr,"C:/Users/pc/Downloads/run_treCopia.csv")

# WSP_DaFr <- read_csv("C:/Users/pc/Desktop/Wolf Sheep Predation experiment2-spreadsheet.csv")
# 
# WSP_DaFr <- as.data.frame(WSP_DaFr)
# 
# row.names(WSP_DaFr) <- WSP_DaFr[,1]
# 
# WSP_DaFr <- WSP_DaFr[,-1]
# 
# str(WSP_DaFr)

print(paste(" run:", WSP_DaFr[1,1]," w:",WSP_DaFr[2,1]," s:",WSP_DaFr[3,1]," s-w-g:",WSP_DaFr[4,1]))

as.matrix(WSP_DaFr) -> WSP_DaFr
agent <- c("Sheep","Wolves","Grass")
color <- c("yellow","darkblue","green")



for (colNum in 1:dim(WSP_DaFr)[2]) {
  plot(0:500, WSP_DaFr[5:505,colNum]
       , main = paste(" run:", WSP_DaFr[1,colNum]," w:",WSP_DaFr[2,colNum]," s:",WSP_DaFr[3,colNum],"\n",agent[WSP_DaFr[4,colNum]])
       , xlab = "time"
       , ylab = "count"
       , type = "l"
       , col = color[WSP_DaFr[4,colNum]]
  )
}

##############
# stdVec <- apply(WSP_DaFr, 1, sd)
# menVec <- apply(WSP_DaFr, 1, mean)
# madVec <- apply(WSP_DaFr, 1, mad)
# mdnVec <- apply(WSP_DaFr, 1, median)
# minVec <- apply(WSP_DaFr, 1, min)
# maxVec <- apply(WSP_DaFr, 1, max)

# # questo aveva senso con le pecore ma....
# smoothScatter(x = rep(0:1000, ncol(WSP_DaFr)), y = as.vector(as.matrix(WSP_DaFr)) )
# points(mdnVec, type = "l", col = "red"
#      #, ylim = c(min(minVec),max(maxVec))
#      )
# points(minVec,type = "l", col = "red")
# points(maxVec,type = "l", col = "red")
# 
# points((mdnVec + madVec), type = "l", col = "green")
# points((menVec + stdVec), type = "l", col = "yellow")
# points((mdnVec - madVec), type = "l", col = "green")
# #ma queste verdi non vanno un gran che 
# #meglio con le stat dei box plot robusti che vedo più sotto
# points((menVec - stdVec), type = "l", col = "yellow")
# points(menVec, type = "l", col = "orange")


# points(WSP_DaFr[,1],type = "l", col = "blue")
# points(WSP_DaFr[,10],type = "l")
# points(WSP_DaFr[,20],type = "l", col = "green")
# points(WSP_DaFr[,30],type = "l", col = "orange")
# points(WSP_DaFr[,40],type = "l", col = "pink")
# points(WSP_DaFr[,50],type = "l", col = "yellow")

# guardata agli andamenti
for (i in 1:14) {
  plot(WSP_DaFr[151:301,i,7], type = "l")
} # al netto di un interpolazione approssimativa sembra che il restnte sia dovuto a una specie di rumore

#proviamo un smoothscatter
for (i in 1:14) {
  smoothScatter(x = rep(1:nrow(WSP_DaFr[,i,])
                        , ncol(WSP_DaFr[,i,]))
                , y = as.vector(WSP_DaFr[,i,]) )
  
} # sembra ci sia qualcosa ...    ma non lo so .. forse meglio abbandonare la compressione sulle varie run


library(robustbase)

mdcVec <- apply(WSP_DaFr, 1, mc) # skiunes
plot(mdcVec)
summary(mdcVec)  

adjbox(as.vector(as.matrix(WSP_DaFr[200,])) ~ rep(1,100))
adjboxStats(as.vector(as.matrix(WSP_DaFr[200,]))) #questo sarebbe da usare e da rendere smooth

# poi derivate.....e le 5 stat del boxplot su queste

# cmq un po come tenere una tazza di latte: 
# quali punti si può tenere perché oscilli senza sversare?!

plot(table(diff(WSP_DaFr[,1])), type = "h")
DF_diff = WSP_DaFr[-1,] - WSP_DaFr[-nrow(WSP_DaFr),]
adjboxStats(as.vector(as.matrix(DF_diff[,1])))$`stats`

# qua un po di studio sulla derivata.... 
DF_diff <- DF_diff[-(1:199),]
robudif <- apply(DF_diff, 2, function(x) return(adjboxStats(x)$`stats`) )
rownames(robudif) <- c("lower_whisker","lower_hinge", "median", "upper_hinge", "upper_whisker")
summary(t(robudif))
plot(DF_diff[-1,1] ~ DF_diff[-length(DF_diff[,1]),1], type = "p")
abline(lm(DF_diff[-1,1] ~ DF_diff[-length(DF_diff[,1]),1]))
summary(lm(DF_diff[-1,1] ~ DF_diff[-length(DF_diff[,1]),1]))
plot(DF_diff[2:10,1] ~ DF_diff[1:9,1], type = "l")
###################

library(robustbase)

WSP_DaFr <- apply(WSP_DaFr,c(1,2,3),as.numeric)

WSP_RobustSt <- apply(WSP_DaFr, c(1,2), function(x) return(c(adjboxStats(x)$`stats`, length(adjboxStats(x)$`out`)))) # 

WSP_Last_wind <- apply(WSP_DaFr[292:301,,], 2, function(x) return(c(mean(x), sd(x), adjboxStats(x)$`stats`)))

dim(WSP_RobustSt)

## nomi variabili

titoli_output = c("tasso disoccupazione"
                  , "PIL log_n"
                  , "max - Production of new firms"
                  , "max - Production of new firms"
                  , "mean - Production of new firms"
                  , "mean - ricchezza lavoratori"
                  , "max - ricchezza lavoratori"
                  , "min - ricchezza lavoratori"
                  , "mean - wage offered Wb"
                  , "min - wage offered Wb"
                  , "max - wage offered Wb"
                  , "mean - Contractual interest rate"
                  , "max - Contractual interest rate"
                  , "min - Contractual interest rate"
)
titoli_output = c(titoli_output
                  , "Production of new firms"
                  , "ricchezza lavoratori"
                  , "wage offered Wb"
                  , "Contractual interest rate"
                  )


# graphing ###########

library(deming)
par(ask = T)

for (i in 1:14) {
  plot(WSP_RobustSt[3,,i] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       , xlab = "tick num"
       , ylab = titoli_output[i] #"disoccup"
       , ylim = c(min(WSP_RobustSt[1,,i]),max(WSP_RobustSt[5,,i]))
       , type = "l"
       , col = "red")
  LMsumm <- pbreg(I(WSP_RobustSt[3,151:301,i]) ~ I(151:301))
  #pear <- cor.test(WSP_RobustSt[3,151:301,i], 151:301)
  abline(LMsumm
         , lty = 3)
  abline(v = 151
         , lty = 3)
  #title(main = paste("RSE:",summary(LMsumm)$sigma))
  title(paste(" Passing-Bablock Regression in [151,301]:\n slope:"
              , signif(LMsumm$coefficients[2], digits = 3)
              , "    "
              , "intercept at [151]:"
              , signif(LMsumm$coefficients[2] * 151 + LMsumm$coefficients[1]
                       #,LMsumm$fitted.values[151]
                       , digits = 3)
              , "    "
              , "MAE:"
              , signif(mean(abs(LMsumm$residuals))
                       #, summary(LMsumm)$sigma
                       , digits = 3)
              , "\n"
              , "mean of outlier ratio on y:"
              , signif(mean(WSP_RobustSt[6,151:301,i]/1000)
                       , digits = 3)
              # , "                             "
              # , "r_pearson:"
              # , signif(pear$estimate, 3)
              # , "al 95% tra:"
              # , paste(signif(pear$conf.int, 3), collapse = " e ")
              )
        # , sub = paste("Last 10 step window:\n mean:"
        #               , signif(WSP_Last_wind[1,i], 4)
        #               , ";  sd:"
        #               , signif(WSP_Last_wind[2,i], 4)
        #               , "\n median:"
        #               , signif(WSP_Last_wind[5,i], 4)
        #               , ";  lower hinge diff:"
        #               , signif(WSP_Last_wind[4,i] - WSP_Last_wind[5,i], 4)
        #               , ";  upper hinge diff:"
        #               , signif(WSP_Last_wind[6,i] - WSP_Last_wind[5,i], 4)
        #               )
        , cex.main = 0.75, adj = 0#,   font.main = 1, col.main = "dark red"
        #, cex.sub = 0.75, adj = 1, font.sub = 3, col.sub = "dark red"
        )
  title(sub = paste("Last 10 step window:\n mean:"
                    , signif(WSP_Last_wind[1,i], 4)
                    , ";    sd:"
                    , signif(WSP_Last_wind[2,i], 4)
                    , "\n median:"
                    , signif(WSP_Last_wind[5,i], 4)
                    , "                   \n   lower hinge diff:"
                    , signif(WSP_Last_wind[4,i] - WSP_Last_wind[5,i], 4)
                    , ";    upper hinge diff:"
                    , signif(WSP_Last_wind[6,i] - WSP_Last_wind[5,i], 4)
                    , " "
                    )
        , cex.sub = 0.75, adj = 1, font.sub = 4, col.sub = "dark red"
        ) 
  
  points(WSP_RobustSt[1,,i] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"1
       #, ylab = "disoccup"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "blue")
  points(WSP_RobustSt[5,,i] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"
       #, ylab = "disoccup"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "blue")
# plot(WSP_RobustSt[6,,2] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
#      , xlab = "tick num"
#      , ylab = "outlayers"
#      #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
#      , type = "l"
#      , col = "green")
  points(WSP_RobustSt[2,,i] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"
       #, ylab = "disoccup"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "purple")
  points(WSP_RobustSt[4,,i] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"
       #, ylab = "disoccup"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "purple")
  points(WSP_RobustSt[3,,i] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
     #, xlab = "tick num"
     #, ylab = paste("var: ",i) #"disoccup"
     #, ylim = c(min(WSP_RobustSt[1,,i]),max(WSP_RobustSt[5,,i]))
     , type = "l"
     , col = "red")
}

for (i in 1) {
  # Production of new firms ####
plot(WSP_RobustSt[3,,5] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
     , xlab = "tick num"
     , ylab = "Production of new firms"
     , ylim = c(min(WSP_RobustSt[3,,3]),max(WSP_RobustSt[3,,4]))
     , type = "l"
     , col = "orange")
abline(v = 150
       , lty = 3)
points(WSP_RobustSt[3,,3] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"
       #, ylab = "ricchezza_lavorat_media"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "green")
points(WSP_RobustSt[3,,4] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"
       #, ylab = "ricchezza_lavorat_media"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "cyan")  # qui due si ripetono nei dati


# ricchezza lavoratori ####
plot(WSP_RobustSt[3,,6] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
     , xlab = "tick num"
     , ylab = "ricchezza lavoratori"
     , ylim = c(min(WSP_RobustSt[3,,8]),max(WSP_RobustSt[3,,7]))
     , type = "l"
     , col = "orange")
abline(v = 150
       , lty = 3)
points(WSP_RobustSt[3,,8] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"
       #, ylab = "ricchezza_lavorat_media"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "green")
points(WSP_RobustSt[3,,7] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"
       #, ylab = "ricchezza_lavorat_media"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "cyan")

# wage offered Wb ####
plot(WSP_RobustSt[3,,9] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
     , xlab = "tick num"
     , ylab = "wage offered Wb"
     , ylim = c(min(WSP_RobustSt[3,,10]),max(WSP_RobustSt[3,,11]))
     , type = "l"
     , col = "orange")
abline(v = 150
       , lty = 3)
points(WSP_RobustSt[3,,10] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"
       #, ylab = "ricchezza_lavorat_media"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "green")
points(WSP_RobustSt[3,,11] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"
       #, ylab = "ricchezza_lavorat_media"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "cyan")




# Contractual interest rate ####
plot(WSP_RobustSt[3,,12] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
     , xlab = "tick num"
     , ylab = "Contractual interest rate"
     , ylim = c(min(WSP_RobustSt[3,,14]),max(WSP_RobustSt[3,,13]))
     , type = "l"
     , col = "orange")
abline(v = 150
       , lty = 3)
points(WSP_RobustSt[3,,14] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"
       #, ylab = "ricchezza_lavorat_media"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "green")
points(WSP_RobustSt[3,,13] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
       #, xlab = "tick num"
       #, ylab = "ricchezza_lavorat_media"
       #, ylim = c(min(WSP_RobustSt[1,,1]),max(WSP_RobustSt[5,,1]))
       , type = "l"
       , col = "cyan")
# points(WSP_RobustSt[3,,12] ~ I(0:(dim(WSP_RobustSt)[2] - 1))
#      , type = "l"
#      , col = "orange"
#      , lty = 3)

}

#####



##### export grafici    #####

# 
# plots.dir.path <- list.files(tempdir(), pattern = "rs-graphics", full.names = TRUE)
# plots.png.paths <- list.files(plots.dir.path, pattern = ".png", full.names = TRUE)
# 
# plots.png.detials <- file.info(plots.png.paths)
# plots.png.detials <- plots.png.detials[order(plots.png.detials$mtime),]
# sorted.png.names <- gsub(plots.dir.path, "path_to_your_dir", row.names(plots.png.detials), fixed=TRUE)
# numbered.png.names <- paste0("path_to_your_dir/", 1:length(sorted.png.names), ".png")
# 
# file.copy(from = plots.png.paths, to = "C:/Users/pc/Downloads/run_tre.csv")


# Rename all the .png files as: 1.png, 2.png, 3.png, and so on.
file.rename(from=sorted.png.names, to=numbered.png.names)



#####
plot((WSP_RobustSt[6,,6]/1000) ~ I(0:(dim(WSP_RobustSt)[2] - 1))
     , xlab = "tick num"
     , ylab = "rate of outlayer"
     , main = "riccchezza lavoratori"
     #, ylim = c(min(WSP_RobustSt[1,,6]),max(WSP_RobustSt[5,,6]))
     , type = "l"
     , col = "orange")


WSP_RobustSt_AN <- WSP_RobustSt[,-(1:150),]

for (i in 1:14) {
  try(LMsumm <- summary(
    lm(I(WSP_RobustSt_AN[3,,i]) ~ I(0:(dim(WSP_RobustSt_AN)[2] - 1)))
    ))
  print(LMsumm$adj.r.squared)
}


# rnn package  ####

library(rnn)
library(nnet)
# trainingY = array(scale(
#   t(WSP_DaFr[-(1:151),1,1:700]))
#   , dim = dim(t(WSP_DaFr[-(1:151),1,1:700])))
trainingY = int2bin(t(WSP_DaFr[-(1:151),1,1:700]) * 1000 - 3
                    , length = 8)
trainingY = array(trainingY, dim = c(dim(t(WSP_DaFr[-(1:151),1,1:700])),8))
# testY = array(scale(
#   t(WSP_DaFr[-(1:151),1,701:1000]))
#   , dim = dim(t(WSP_DaFr[-(1:151),1,701:1000])))
testY = int2bin(t(WSP_DaFr[-(1:151),1,701:1000]) * 1000 - 3
                , length = 8)
testY = array(testY, dim = c(dim(t(WSP_DaFr[-(1:151),1,701:1000])),8))
# trainingX = array(scale(
#   t(WSP_DaFr[c(-(1:150),-nrow(WSP_DaFr)),1,1:700]))
#   , dim = dim(t(WSP_DaFr[c(-(1:150),-nrow(WSP_DaFr)),1,1:700])))
trainingX = int2bin(t(WSP_DaFr[c(-(1:150),-nrow(WSP_DaFr)),1,1:700]) * 1000 - 3
                    , length = 8)
trainingX = array(trainingX, dim = c(dim(t(WSP_DaFr[-(1:151),1,1:700])),8))
# testX = array(scale(
#   t(WSP_DaFr[c(-(1:150),-nrow(WSP_DaFr)),1,701:1000]))
#   , dim = dim(t(WSP_DaFr[c(-(1:150),-nrow(WSP_DaFr)),1,701:1000])))
testX = int2bin(t(WSP_DaFr[c(-(1:150),-nrow(WSP_DaFr)),1,701:1000]) * 1000 - 3
                , length = 8)
testX = array(testX, dim = c(dim(t(WSP_DaFr[-(1:151),1,701:1000])),8))


passo_iniz = 30
passi = 10 # +1

modelloRicor <- trainr(
  Y = as.matrix(trainingY[,passo_iniz + passi + 1 - 1,])
  , X = trainingX[,passo_iniz:(passo_iniz + passi),]
  #, model = modelloRicor
  , learningrate = 0.1
  , learningrate_decay = 0.999 # 0.9
  , momentum = 0.2
  , hidden_dim = c(10,10,10)
  , numepochs = 100
  , batch_size = 50
  , use_bias = T
  , network_type = "lstm"
  , seq_to_seq_unsync = T
  #, loss_function = cross_ent
)

plot(colMeans(modelloRicor$error)
     , type = 'l'
     , xlab = 'epoch'
     , ylab = 'errors')
  
predetto = predictr(modelloRicor
                    ,testX[,passo_iniz:(passo_iniz + passi),])
predetto_int = bin2int(predetto)

# plot(as.vector(predetto)[order(testY[,passo_iniz + passi + 1 - 1,])]
#      , ylim = c(min(c(predetto,as.matrix(testY[,passo_iniz + passi + 1 - 1])))
#                 ,max(c(predetto,as.matrix(testY[,passo_iniz + passi + 1 - 1]))))
#      , type = "l")
# points(as.matrix(sort(testY[,passo_iniz + passi + 1 - 1]))
#        , type = "l"
#        , col = "blue")

plot(sort(bin2int(trainingY[,passo_iniz + passi + 1 - 1,])), type = "l", ylim = c(0,256))
points(predetto_int[order(bin2int(trainingY[,passo_iniz + passi + 1 - 1,]))])
