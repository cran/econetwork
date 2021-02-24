elgrin <- function(presence, metaweb, environmentvar, ncores=1){
    if("igraph" %in% class(metaweb)){
        metaweb <- as.matrix(as_adjacency_matrix(metaweb))
    } else{
        metaweb <- as.matrix(metaweb)
    }
    metaweb <- metaweb*1.0 ## requiring numeric weights
    if(nrow(presence)!=nrow(metaweb))
        stop("inconsistent number of species between presence data and metaweb")
    environmentvar <- as.matrix(environmentvar)
    if(ncol(presence)!=nrow(environmentvar))
        stop("inconsistent number of locations between presence data and environmentvar data")
    environmentvar <- apply(environmentvar,2,function(v)(v-mean(v))/sd(v))
    elgrincore(presence, metaweb, environmentvar, ncores)
}
