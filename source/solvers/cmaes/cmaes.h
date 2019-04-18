#ifndef _KORALI_CMAES_H_
#define _KORALI_CMAES_H_

#include "solvers/base/base.h"
#include "parameters/gaussian/gaussian.h"
#include <chrono>
#include <map>

namespace Korali::Solver
{

class CMAES : public Korali::Solver::Base
{
 public:

 // Constructor / Destructor
 CMAES(nlohmann::json& js);
 ~CMAES();

 // Runtime Methods (to be inherited from base class in the future)
 void prepareGeneration();
 bool checkTermination();
 void updateDistribution(const double *fitnessVector);
 void run();
 void processSample(size_t sampleId, double fitness);

 // Serialization Methods
 nlohmann::json getConfiguration();
 void setConfiguration(nlohmann::json& js);
 void setState(nlohmann::json& js);

 private:

 // Korali Runtime Variables
 double* _fitnessVector; /* objective function values [_s] */
 double* _samplePopulation; /* sample coordinates [_s x _k->N] */
 size_t _currentGeneration; /* generation count */
 bool* _initializedSample;
 char _terminationReason[500];
 char filepath[500];

 size_t _finishedSamples;
 size_t _s; /* number of samples per generation */
 size_t _mu; /* number of best samples for mean / cov update */
 std::string _muType; /* linearDecreasing, Equal or Logarithmic */
 double* _muWeights; /* weights for mu best samples */
 double _muEffective;
 double _muCovariance;

 double _sigmaCumulationFactor; /* default calculated from muEffective and dimension */
 double _dampFactor; /* dampening parameter determines controls step size adaption */
 double _cumulativeCovariance; /* default calculated from dimension */
 double _covarianceMatrixLearningRate;

 size_t _diagonalCovarianceMatrixEvalFrequency;
 size_t _covarianceEigenEvalFreq;

 // Stop conditions
 size_t _maxFitnessEvaluations;   // Defines maximum number of fitness evaluations
 double _stopFitnessDiffThreshold; // Defines minimum function value differences before stopping
 double _stopMinDeltaX; // Defines minimum delta of input parameters among generations before it stops.
 double _stopMinFitness; // Defines the minimum fitness allowed, otherwise it stops
 double _stopTolUpXFactor; // Defines the minimum fitness allowed, otherwise it stops
 double _stopCovKond; // Defines the maximal condition number of the covariance matrix
 size_t _maxGenenerations; // Max number of generations.
 std::string _ignorecriteria; /* Termination Criteria(s) to ignore: 
    Fitness Value, Fitness Diff Threshold, Max Standard Deviation, 
    Max Kondition Covariance, No Effect Axis, No Effect Standard Deviation, 
    Max Model Evaluations, Max Generations */

 // Private CMAES-Specific Variables
 double sigma;  /* step size */
 Parameter::Gaussian* _gaussianGenerator;

 double currentBest; /* best ever fitness */
 double prevBest; /* best ever fitness from previous generation */ 
 double *rgxmean; /* mean "parent" */
 double *rgxbestever; /* bestever vector */
 double *curBest; /* holding all fitness values (fitnessvector) */ 
 int *index; /* sorting index of current sample pop (index[0] idx of current best). */
 double currentFunctionValue; /* best fitness current generation */
 double prevFunctionValue; /* best fitness previous generation */

 double **C; /* lower triangular matrix: i>=j for C[i][j] */
 double **B; /* matrix with normalize eigenvectors in columns */
 double *rgD; /* axis lengths */

 double *rgpc; /* evolution path for cov update */
 double *rgps; /* exponential for sigma update */
 double *rgxold; /* mean "parent" previous generation */
 double *rgBDz; /* for B*D*z */
 double *rgdTmp; /* temporary (random) vector used in different places */
 double *rgFuncValue; /* holding all fitness values (fitnessvector) */

 size_t countevals; /* Number of function evaluations */
 size_t countinfeasible; /* Number of samples outside of domain given by bounds */
 double maxdiagC; /* max diagonal element of C */
 double mindiagC; /* min diagonal element of C */
 double maxEW; /* max Eigenwert of C */
 double minEW; /* min Eigenwert of C */

 bool flgEigensysIsUptodate;

 // Private CMA-ES-Specific Methods
 void reSampleSingle(size_t idx);
 void adaptC2(int hsig);
 void updateEigensystem(int flgforce);
 void eigen(size_t N, double **C, double *diag, double **Q) const;
 int maxIdx(const double *rgd, int len) const;
 int minIdx(const double *rgd, int len) const;
 void sorted_index(const double *rgFunVal, int *index, int n) const;
 bool isFeasible(const double *pop) const;
 double doubleRangeMax(const double *rgd, int len) const;
 double doubleRangeMin(const double *rgd, int len) const;
 bool isStoppingCriteriaActive(const char *criteria) const;

 // Print Methods 
 void printGeneration() const;
 void printFinal() const;
};

} // namespace Korali

#endif // _KORALI_CMAES_H_
