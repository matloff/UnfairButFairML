
# in some ML contexts, those who are affected may actually want
# sensitive variables to be used in prediction; here we investigate
# the desirability of this approach

# args

#   data: data frame; 
#   yName: name of "Y"; if dichtomous, to be predicted
#   sName: name of "S", the sensitive var. (only 1 allowed); 
#      "S" must be a factor
#   qeFtnName: name of the qe-series ML predictive function; 
#      currently, only default args can be used

UBF <- function(data,yName,sName,qeFtnName) 
{

   # preliminary checks and preparation
   require(qeML)
   # 'data' must be a data frame
   qeML:::checkForNonDF(data)
   # change character variables to factors, especially "Y"
   for (i in 1:ncol(data)) 
      if (is.character(data[,i])) data[,i] <- as.factor(data[,i])
   yCol <- which(names(data) == yName)
   allY <- data[,yCol]  # Y for full data set
   if (is.factor(allY) && length(levels(allY)) != 2)
      stop('in classification cases, only 2-class case is allowed')
   classif <- is.factor(allY) 
   sCol <- which(names(data) == sName)
   allS <- data[,sCol]  # S for full data set
   if (!is.factor(allS)) stop('S must be a factor')
   sLevels <- levels(allS)
   
   # fit on full data, with S
   cmd <- paste0(qeFtnName,'(data,yName)')
   overallFitWithS <- runCmd(cmd)

   # fit on full data, with no S (but S-related vars. ARE allowed)
   dataNoS <- data[,-sCol]
   cmd <- paste0(qeFtnName,'(dataNoS,yName)')
   overallFitWithNoS <- runCmd(cmd)

   # get the "X" data, overall and for each S class
   allX <- data[,-yCol]  # X for full dataset, with S
   allXnoS <- data[,-c(yCol,sCol)]  # X for full dataset, with no S
   # row numbers in 'data' for each S class
   rowNumsEachClass <- lapply(sLevels,function(slvl) which(allS == slvl))
   names(rowNumsEachClass) <- sLevels

   # start building output object
   res <- list()
   res$overallBaseAcc <- overallFitWithS$baseAcc
   res$overallAccWithS <- overallFitWithS$testAcc
   res$overallAccWithNoS <- overallFitWithNoS$testAcc

   # will store test-set accuracy of individual fit within each S class,
   # no use of fits to overall data
   MLrunWithinClassTestAcc <- list()

   # will store test-set accuracies arising when each S class is
   # predicted using fit on overall data, with/without S
   predAccsEachClassUsingOverallWithS <- list()
   predAccsEachClassUsingOverallWithNoS <- list()

   # calculate MLrunWithinClassTestAcc, predAccsEachClassUsingOverallWithS,
   # predAccsEachClassUsingOverallWithNoS
   for (slvl in sLevels) {

      rowNums <- rowNumsEachClass[[slvl]]
      
      # run the ML function on this S class only, no overall fit
      thisData <- data[rowNums,-sCol]
      # prep qe-series function call
      cmd <- paste0(qeFtnName,'(thisData,yName)')
      tmp <- runCmd(cmd)
      MLrunWithinClassTestAcc[[slvl]] <- tmp$testAcc

      # find the prediction accuracy for this class if 
      # the overall fit is sued, with/without S
      tmp <- allX[rowNumsEachClass[[slvl]],]
      thisData <- tmp
      tmp <- predict(overallFitWithS,thisData)
      predAccsEachClassUsingOverallWithS[[slvl]] <- 
         if(classif) mean(tmp$predClasses != data[rowNums,yCol]) else
         mean(abs(tmp - data[rowNums,yCol])) 

      tmp <- allXnoS[rowNumsEachClass[[slvl]],]
      thisData <- tmp
      tmp <- predict(overallFitWithNoS,thisData)
      predAccsEachClassUsingOverallWithNoS[[slvl]] <- 
         if(classif) mean(tmp$predClasses != data[rowNums,yCol]) else
         mean(abs(tmp - data[rowNums,yCol])) 

   }

   res$MLrunWithinClassTestAcc <- unlist(MLrunWithinClassTestAcc)
   res$predAccsEachClassUsingOverallWithS <-
      unlist(predAccsEachClassUsingOverallWithS)
   res$predAccsEachClassUsingOverallWithNoS <-
      unlist(predAccsEachClassUsingOverallWithNoS)

   res
}

runCmd <- function(cmd) eval(parse(text=cmd),parent.frame())

