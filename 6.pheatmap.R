
#29.热力图作图 
setwd("C:\\Users\\admin\\Desktop\\zzy\\lung ML")
DEG_exp<-read.table("heatmap.txt",sep='\t',header=T,row.names=1)
designNC =  c( rep("LUDA",519), rep("LUSC",497))
group_info <- data.frame(row.names=names(DEG_exp),groups=designNC)

library(pheatmap)

pdf(file="pheatmap.pdf",width=20,height=10)
pheatmap(DEG_exp,color=colorRampPalette(c("green","black","red"))(100),fontsize_row=10
         ,fontsize_col=10,scale="row",border_color=NA,cluster_col = FALSE,annotation_col=group_info)
dev.off()

#转python进行预后分析

