\name{divPartition}
\alias{divPartition}
\title{
Partitionning network diversity in alpha, beta and gamma diversity
}
\description{
 This function computes alpha, beta and gamma diversity of a list of networks. It measures either group, links, or probability of links diversity. 
}
\usage{
divPartition(gList, groups, eta=1, framework=c('RLC','Chao'), type=c('P','L','Pi'))
}
\arguments{
  \item{gList}{A \code{list} of graph objects of class
    \code{igraph}.}
  \item{groups}{A named vector of class \code{character}  indicating the group to
    which each node belongs to. The length of \code{groups} must correspond to the number of different nodes present in \code{gList}. The names \code{names(groups)} must
    correspond to the nodes names in \code{gList}. If NULL, the groups are the initial nodes.}
  \item{eta}{A positive number that controls the weight given to abundant groups/links. Default value is 1.}
  \item{framework}{The framework used to partitionate diversity, either Reeve Leinster Cobbold ('RLC') or Chao ('Chao')}
  \item{type}{The type of diversity to measure and partitionate. It can be groups diversity ('P'), link diversity ('L') or probability of link diversity ('Pi').}
}
\value{
Returns a \code{list} the following components:
\item{mAlpha}{The mean value of alpha-diversity accross all networks.}
\item{Alphas}{A vector of \code{numeric} containing the local alpha-diversities (i.e. the alpha-diversity value for each network).}
\item{Beta}{The value of the overall beta-diversity}
\item{Gamma}{The value of the gamma-diversity}
}
\references{
Marc Ohlmann, Vincent Miele, Stephane Dray, Loic Chalmandrier, Louise O'Connor & Wilfried Thuiller, Diversity indices for ecological networks: a unifying framework using Hill numbers. Ecology Letters (2019) <doi:10.1111/ele.13221>
}
\author{
Authors: Stephane Dray, Vincent Miele, Marc Ohlmann, Wilfried Thuiller
Maintainer: Wilfried Thuiller <wilfried.thuiller@univ-grenoble-alpes.fr>
}
\examples{
# Generating a set of Erdos-Renyi graphs and give name to nodes
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

# Measure of link diversity
divPartition(gList, groups, framework='Chao', type = 'L') 
}