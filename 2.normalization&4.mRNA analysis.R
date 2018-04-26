
#接在1.processing后面
# 3.set the working directory
library(impute)
library("wateRmelon")

# 4.set the working directory
setwd("C:\\Users\\admin\\Desktop\\zzy\\lung ML")
Data_LUAD <- read.table("LUAD_mRNA.txt",header = T,row.names = 1,sep = "\t")
Data_LUSC <- read.table("LUSC_mRNA.txt",header = T,row.names = 1,sep = "\t")
#match.name <- read.table("TCGA_mRNA_probes.txt", header = T, sep="\t", quote="",row.names=1)
mean_LuAD <- mean(Data_LUAD)
probe_exp <- cbind(Data_LUAD,Data_LUSC)
# 5.use the geneid matrix to do analysis
mat=as.matrix(probe_exp)
mat=impute.knn(mat)
matData=mat$data
matData=matData+0.00001

# 6.normalization
matData=matData[rowMeans(matData)>0.005,]
matData = betaqn(matData)
write.table(matData,file="norm.txt",sep="\t",quote=F)

#### 7. Use the log.pl then limma
library(limma)
eset<-read.table("log_norm.txt",sep='\t',header=T,row.names=1)

# 8.change the sample number
condition=factor(c(rep("LUAD",519),rep("LUSC",497)))
design<-model.matrix(~-1+condition)
colnames(design)<-c("LUAD","LUSC")
contranst.matrix<-makeContrasts(contrasts="LUSC-LUAD",levels=design)
fit<-lmFit(eset,design)
fit1<-contrasts.fit(fit,contranst.matrix)
fit2<-eBayes(fit1)
dif<-topTable(fit2,coef="LUSC-LUAD",n=nrow(fit2),adjust="BH")
genesymbol<-rownames(dif)
dif<-cbind(genesymbol,dif)
write.table(dif,file="probeid.Foldchange.txt",sep='\t',quote=F,row.names=F)

# 9.Plot volcano
pdf(file="Volcano.pdf")
yMax=max(-log10(dif$adj.P.Val))
xMax=max(abs(dif$logFC))
plot(dif$logFC,-log10(dif$adj.P.Val), xlab="log2(FC)",ylab="-log10(adj.P.Val)",main="Volcano", xlim=c(-xMax,xMax),ylim=c(0,yMax),yaxs="i",pch=20, cex=0.4,col="grey")
diffSub1=subset(dif, dif$adj.P.Val<0.05 & dif$logFC>1)
diffSub2=subset(dif, dif$adj.P.Val<0.05 & dif$logFC<(-1))
points(diffSub1$logFC,-log10(diffSub1$adj.P.Val), pch=20, col="red",cex=0.4)
points(diffSub2$logFC,-log10(diffSub2$adj.P.Val), pch=20, col="green",cex=0.4)
cut = -log10(0.05)
abline(h=cut,lty=2,lwd=1,col="blue")
abline(v=c(-log2(2),log2(2)),lty=2,lwd=1,col="blue")
dev.off()


# 10.set the foldchange value in the bracket
dif2<-topTable(fit2,coef="LUSC-LUAD",n=nrow(fit2),lfc=log2(2),adjust="BH")

# 11.set the adj.P.Val in the bracket
dif2<-dif2[dif2[,"adj.P.Val"]<0.05,]
genesymbol<-rownames(dif2)
dif2<-cbind(genesymbol,dif2)
dif2<-dif2[order(dif2$logFC),]
write.table(dif2,file="diff.probeid.FDR0.05.txt",sep='\t',quote=F,row.names=F)

dif2$genesymbol<-rownames(dif2)
loc<-match(dif2$genesymbol,rownames(eset))
DEG_exp<-eset[loc,]
genesymbol<-rownames(DEG_exp)
DEG_exp<-cbind(genesymbol,DEG_exp)
write.table(DEG_exp,file="probeid.FDR0.05.exprs.txt",sep='\t',quote=F,row.names=F)

