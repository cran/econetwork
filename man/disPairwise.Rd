\name{disPairwise}
\alias{disPairwise}
\title{
Computation of the dissimilarity matrix (pairwise beta-diversity) for a set of networks
}
\description{
  Computation of the dissimilarity matrix for a set of networks. Each value of the matrix is the pairwise beta-diversity, computed using Hill numbers. It measures the dissimilarity in terms of groups, links, or probability of links respectively.  
}
\usage{
disPairwise(gList, groups=NULL, eta=1, type=c('P','L','Pi'))
}
\arguments{
  \item{gList}{A \code{list} of graph objects of class
    \code{igraph}.}
  \item{groups}{A named vector of class \code{character}  indicating the group to
    which each node belongs to. The length of \code{groups} must correspond to the number of different nodes present in \code{gList}. The names \code{names(groups)} must
    correspond to the nodes names in \code{gList}. If NULL, the groups are the initial nodes.}
  \item{eta}{A positive number that controls the weight given to abundant groups/links. Default value is 1.}
  \item{type}{The type of diversity used to measure dissimilarity. It can be groups diversity ('P'), links diversity ('L') or probability of links diversity ('Pi').}
}
\value{
  Return a \code{matrix} whose elements are the pairwise dissimilarities. 
}
\references{
Marc Ohlmann, Vincent Miele, Stephane Dray, Loic Chalmandrier, Louise O'Connor & Wilfried Thuiller, Diversity indices for ecological networks: a unifying framework using Hill numbers. Ecology Letters (2019) <doi:10.1111/ele.13221>
}
\author{
Authors: Stephane Dray, Vincent Miele, Marc Ohlmann, Wilfried Thuiller
Maintainer: Wilfried Thuiller <wilfried.thuiller@univ-grenoble-alpes.fr>
}
\examples{
# Generating a set of Erdos-Renyi graphs and give name to nodes.
library(igraph)
nbGraph = 10
gList = c()
n = 57 # number of nodes of each graph
C = 0.1  # connectance of each graph
for(i in 1:nbGraph){
  graphLocal = erdos.renyi.game(n, type='gnp', p.or.m =C, directed=TRUE)
  V(graphLocal)$name = as.character(1:57)
  gList = c(gList,list(graphLocal))
}

groups = c(rep("a",23),rep("b",34)) # vector that gives the group of each node
names(groups) = as.character(1:57)

# Dissimilarity matrix based on links beta-diversity
disPairwise(gList, groups, type = 'L') 
}
