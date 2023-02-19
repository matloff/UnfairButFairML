
# args

#   data: data frame; 
#   yName: name of "Y"; if dichtomous, must be a factor
#   sName: name of "S"; "S" must be a factor, currently dichtomous
#   qeFtnName: name of the qe-series ML function; currently, default args

UBF <- function(data,yName,sName,qeFtnName) 
{
   cmd <- paste0(qeFtnName,'(data,yName)')
   overallFitWithS <- runCmd(cmd)

   sCol <- which(names(data) == sName)
   dataNoS <- data[,-sCol]
   cmd <- paste0(qeFtnName,'(dataNoS,yName)')
   overallFitWithNoS <- runCmd(cmd)

   allS <- data[,sCol]
   sLevels <- levels(allS)
   yCol <- which(names(data) == yName)
   allX <- data[,-yCol]
   allXnoS <- data[,-c(yCol,sCol)]
   rowNumsEachClass <- list()  # row numbers in 'data' for each class
   for (slvl in sLevels) {
      rowNums <- which(allS == slvl)
      rowNumsEachClass[[slvl]] <- rowNums
   }

   # res <- list(data,yName,sName,qeFtnName)
   res <- list()
   res$overallBaseAcc <- overallFitWithS$baseAcc
   res$overallAccWithS <- overallFitWithS$testAcc
   res$overallAccWithNoS <- overallFitWithNoS$testAcc
   # now fit within each class, in various contexts
   MLrunWithinClassTestAcc <- list()
   predAccsEachClassUsingOverallWithS <- list()
   predAccsEachClassUsingOverallWithNoS <- list()
   for (slvl in sLevels) {
      rowNums <- rowNumsEachClass[[slvl]]
      thisData <- data[rowNums,]
      thisData <- thisData[,-sCol]
      cmd <- paste0(qeFtnName,'(thisData,yName)')
      tmp <- runCmd(cmd)
      MLrunWithinClassTestAcc[[slvl]] <- tmp$testAcc
      tmp <- allX[rowNumsEachClass[[slvl]],]
      thisData <- tmp
      tmp <- predict(overallFitWithS,thisData)
      predAccsEachClassUsingOverallWithS[[slvl]] <- 
         mean(tmp$predClasses != data[rowNums,yCol])
      tmp <- allX[rowNumsEachClass[[slvl]],-sCol]
      thisData <- tmp
      tmp <- predict(overallFitWithNoS,thisData)
      predAccsEachClassUsingOverallWithNoS[[slvl]] <- 
         mean(tmp$predClasses != data[rowNums,yCol])
   }
   res$MLrunWithinClassTestAcc <- unlist(MLrunWithinClassTestAcc)
   res$predAccsEachClassUsingOverallWithS <-
      unlist(predAccsEachClassUsingOverallWithS)
   res$predAccsEachClassUsingOverallWithNoS <-
      unlist(predAccsEachClassUsingOverallWithNoS)

   res
}

runCmd <- function(cmd) eval(parse(text=cmd),parent.frame())

