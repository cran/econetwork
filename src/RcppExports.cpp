// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <RcppGSL.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// elgrincore
List elgrincore(IntegerMatrix YR, NumericMatrix AmetaR, NumericMatrix EnvironmentR, int nbthreads);
RcppExport SEXP _econetwork_elgrincore(SEXP YRSEXP, SEXP AmetaRSEXP, SEXP EnvironmentRSEXP, SEXP nbthreadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerMatrix >::type YR(YRSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type AmetaR(AmetaRSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type EnvironmentR(EnvironmentRSEXP);
    Rcpp::traits::input_parameter< int >::type nbthreads(nbthreadsSEXP);
    rcpp_result_gen = Rcpp::wrap(elgrincore(YR, AmetaR, EnvironmentR, nbthreads));
    return rcpp_result_gen;
END_RCPP
}
// elgrinsimcore
IntegerMatrix elgrinsimcore(NumericMatrix AmetaR, NumericMatrix EnvironmentR, NumericVector aR, NumericVector alR, NumericMatrix bR, NumericMatrix cR, NumericVector betaR, NumericVector betaabsR, IntegerMatrix compatR, int nbthreads);
RcppExport SEXP _econetwork_elgrinsimcore(SEXP AmetaRSEXP, SEXP EnvironmentRSEXP, SEXP aRSEXP, SEXP alRSEXP, SEXP bRSEXP, SEXP cRSEXP, SEXP betaRSEXP, SEXP betaabsRSEXP, SEXP compatRSEXP, SEXP nbthreadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type AmetaR(AmetaRSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type EnvironmentR(EnvironmentRSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type aR(aRSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type alR(alRSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type bR(bRSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type cR(cRSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type betaR(betaRSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type betaabsR(betaabsRSEXP);
    Rcpp::traits::input_parameter< IntegerMatrix >::type compatR(compatRSEXP);
    Rcpp::traits::input_parameter< int >::type nbthreads(nbthreadsSEXP);
    rcpp_result_gen = Rcpp::wrap(elgrinsimcore(AmetaR, EnvironmentR, aR, alR, bR, cR, betaR, betaabsR, compatR, nbthreads));
    return rcpp_result_gen;
END_RCPP
}
