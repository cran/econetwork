/*
* This file is part of econetwork
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>
*/
#include<Rcpp.h>
#include <EltonModel.h>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <unistd.h>
#ifdef _OPENMP
#include<omp.h>
#endif


using namespace econetwork;
using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]
List elgrincore(IntegerMatrix YR, NumericMatrix AmetaR, NumericMatrix EnvironmentR, int nbthreads = 1){
#ifdef _OPENMP
  omp_set_num_threads(nbthreads);
#endif
  Eigen::initParallel();
  
  auto nbSpecies = AmetaR.nrow();
  auto nbLocations = YR.ncol();
  auto nbCovariates = EnvironmentR.ncol();
  
  // Ameta
  //double* AmetaR = new double[nbSpecies*nbSpecies]
  // Y
  Eigen::MatrixXd Y(nbSpecies, nbLocations);
  for(auto i=0; i<nbSpecies; i++)
    for(auto l=0; l<nbLocations; l++)
      Y(i,l) = YR(i,l);
#ifdef VERBOSE
  cout<<"# Y data"<<endl<<Y.mean()<<endl;
#endif
  // Environment
  //double* EnvironmentR = new double[nbLocations*nbCovariates];
  
  std::shared_ptr<EnvironmentEffect> peffecti = std::make_shared<EnvironmentEffect>(nbSpecies,nbLocations,nbCovariates);
  peffecti->loadEnvironmentData(EnvironmentR.begin());
  
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  double alphainit = 1+log(Y.mean()/(1-Y.mean())); // assuming beta=0 then Y.mean=exp(alpha)/1-exp(alpha))
  double betainit = 0.;
  EltonModel modeli(nbSpecies,nbLocations,alphainit,betainit,1.,SamplingType::speciesDependent);
  //modeli.loadEpsilon(&epsilonR[0][0]);
  modeli.loadAmeta(AmetaR.begin()); 
  modeli.loadEnvironment(peffecti);
  modeli.simulateX(Y,true);
#ifdef VERBOSE
  cout<<"# X simulated"<<endl<<modeli.getX().mean()<<endl;
  cout.precision(6);
  cout<<"# Q2 before alpha update"<<endl<<modeli.getQ2()<<endl;
#endif
  modeli.updateAlphaBeta();
#ifdef VERBOSE
  cout.precision(6);
  cout<<"# Q2"<<endl<<modeli.getQ2()<<endl;
  cout.precision(3);
  cout<<"# alphaS estimated"<<endl<<modeli.getAlphaSpecies().transpose()<<endl;
  cout<<"# alphaL estimated"<<endl<<modeli.getAlphaLocations().transpose()<<endl;
  cout<<"# beta estimated"<<endl<<modeli.getBeta().transpose()<<endl;
  cout<<"# betaabs estimated"<<endl<<modeli.getBetaAbs().transpose()<<endl;
  cout<<"# coeffa estimated"<<endl<<peffecti->getCoefficientA().transpose()<<endl;
  cout<<"# coeffb estimated"<<endl<<peffecti->getCoefficientB().transpose()<<endl;
#endif
  Rcpp::NumericVector ai(nbSpecies);
  Rcpp::NumericMatrix bi(nbSpecies,nbCovariates), ci(nbSpecies,nbCovariates);
  for(unsigned int i=0; i<nbSpecies; i++){
    ai[i] = modeli.getAlphaSpecies()[i];
    for (unsigned int k=0; k<nbCovariates; k++){
      bi(i,k) = peffecti->getCoefficientA()(i,k);    
      ci(i,k) = peffecti->getCoefficientB()(i,k);
    }
  }
  Rcpp::NumericVector al(nbLocations), betap(nbLocations), betaa(nbLocations);
  for(unsigned int l=0; l<nbLocations; l++){
    al[l] = modeli.getAlphaLocations()[l];
    betap[l] = modeli.getBeta()[l];
    betaa[l] = modeli.getBetaAbs()[l];    
  }
  
  return List::create(Rcpp::Named("a") = ai,
		      Rcpp::Named("al") = al,
		      Rcpp::Named("b") = bi,
		      Rcpp::Named("c") = ci,
		      Rcpp::Named("betaPres") = betap,
		      Rcpp::Named("betaAbs") = betaa);
}
