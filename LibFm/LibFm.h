// LibFm.h

#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <iterator>
#include <algorithm>
#include <iomanip>
#include <vector>
#include "util/util.h"
#include "util/cmdline.h"
#include "fm_core/fm_model.h"
#include "libfm/src/Data.h"
#include "libfm/src/fm_learn.h"
#include "libfm/src/fm_learn_sgd.h"
#include "libfm/src/fm_learn_sgd_element.h"
#include "libfm/src/fm_learn_sgd_element_adapt_reg.h"
#include "libfm/src/fm_learn_mcmc_simultaneous.h"

using namespace System;

namespace LibFm {

	public ref class LibFmManager
	{
	private:
		fm_model *fm;
		FmData *train;
		fm_learn* fml;

	public:
		
		void Train(int argc, char **params)
		{
			try {
				CMDLine cmdline(argc, params);
				const std::string param_task = cmdline.registerParameter("task", "r=regression, c=binary classification [MANDATORY]");
				const std::string param_train_file = cmdline.registerParameter("train", "filename for training data [MANDATORY]");
				const std::string param_test_file = cmdline.registerParameter("test", "filename for test data [MANDATORY]");

				const std::string param_dim = cmdline.registerParameter("dim", "'k0,k1,k2': k0=use bias, k1=use 1-way interactions, k2=dim of 2-way interactions; default=1,1,8");
				const std::string param_regular = cmdline.registerParameter("regular", "'r0,r1,r2' for SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way regularization");
				const std::string param_init_stdev = cmdline.registerParameter("init_stdev", "stdev for initialization of 2-way factors; default=0.1");
				const std::string param_num_iter = cmdline.registerParameter("iter", "number of iterations; default=100");
				const std::string param_learn_rate = cmdline.registerParameter("learn_rate", "learn_rate for SGD; default=0.1");

				const std::string param_method = cmdline.registerParameter("method", "learning method (SGD, SGDA, ALS, MCMC); default=MCMC");

				const std::string param_do_sampling = "do_sampling";
				const std::string param_do_multilevel = "do_multilevel";
				const std::string param_num_eval_cases = "num_eval_cases";

				// Seed
				//long int seed = cmdline.getValue(param_seed, time(NULL));
				srand(time(NULL));

				if (!cmdline.hasParameter(param_method)) { cmdline.setValue(param_method, "mcmc"); }
				if (!cmdline.hasParameter(param_init_stdev)) { cmdline.setValue(param_init_stdev, "0.1"); }
				if (!cmdline.hasParameter(param_dim)) { cmdline.setValue(param_dim, "1,1,8"); }

				if (!cmdline.getValue(param_method).compare("als")) { // als is an mcmc without sampling and hyperparameter inference
					cmdline.setValue(param_method, "mcmc");
					if (!cmdline.hasParameter(param_do_sampling)) { cmdline.setValue(param_do_sampling, "0"); }
					if (!cmdline.hasParameter(param_do_multilevel)) { cmdline.setValue(param_do_multilevel, "0"); }
				}

				// (1) Load the data
				std::cout << "Loading train...\t" << std::endl;
				FmData train(0,
					!(!cmdline.getValue(param_method).compare("mcmc")), // no original data for mcmc
					!(!cmdline.getValue(param_method).compare("sgd") || !cmdline.getValue(param_method).compare("sgda")) // no transpose data for sgd, sgda
					);
				train.load(cmdline.getValue(param_train_file));

				std::cout << "Loading test... \t" << std::endl;
				FmData test(0,
					!(!cmdline.getValue(param_method).compare("mcmc")), // no original data for mcmc
					!(!cmdline.getValue(param_method).compare("sgd") || !cmdline.getValue(param_method).compare("sgda")) // no transpose data for sgd, sgda
					);
				test.load(cmdline.getValue(param_test_file));

				fm = new fm_model();
				// (2) Setup the factorization machine
				{
					//fm->num_attribute = num_all_attribute;
					fm->init_stdev = cmdline.getValue(param_init_stdev, 0.1);
					// set the number of dimensions in the factorization
					{
						std::vector<int> dim = cmdline.getIntValues(param_dim);
						assert(dim.size() == 3);
						fm->k0 = dim[0] != 0;
						fm->k1 = dim[1] != 0;
						fm->num_factor = dim[2];
					}
					fm->init();

				}



				// (3) Setup the learning method:
				if (!cmdline.getValue(param_method).compare("sgd")) {
					fml = new fm_learn_sgd_element();
					((fm_learn_sgd*)fml)->num_iter = cmdline.getValue(param_num_iter, 100);

				}
				else if (!cmdline.getValue(param_method).compare("mcmc")) {
					fm->w.init_normal(fm->init_mean, fm->init_stdev);
					fml = new fm_learn_mcmc_simultaneous();
					//fml->validation = validation;
					((fm_learn_mcmc*)fml)->num_iter = cmdline.getValue(param_num_iter, 100);
					((fm_learn_mcmc*)fml)->num_eval_cases = cmdline.getValue(param_num_eval_cases, test.num_cases);

					((fm_learn_mcmc*)fml)->do_sample = cmdline.getValue(param_do_sampling, true);
					((fm_learn_mcmc*)fml)->do_multilevel = cmdline.getValue(param_do_multilevel, true);
				}
				else {
					throw "unknown method";
				}
				fml->fm = fm;
				fml->max_target = train.max_target; //TODO
				fml->min_target = train.min_target;	//TODO
				//fml->meta = &meta;
				if (!cmdline.getValue("task").compare("r")) {
					fml->task = 0;
				}
				else if (!cmdline.getValue("task").compare("c")) {
					fml->task = 1;
					for (uint i = 0; i < train.target.dim; i++) { if (train.target(i) <= 0.0) { train.target(i) = -1.0; } else { train.target(i) = 1.0; } }
					for (uint i = 0; i < test.target.dim; i++) { if (test.target(i) <= 0.0) { test.target(i) = -1.0; } else { test.target(i) = 1.0; } }
				}
				else {
					throw "unknown task";
				}


				fml->init();
				if (!cmdline.getValue(param_method).compare("mcmc")) {
					// set the regularization; for als and mcmc this can be individual per group
						{
							std::vector<double> reg = cmdline.getDblValues(param_regular);
							assert((reg.size() == 0) || (reg.size() == 1) || (reg.size() == 3) || (reg.size() == (1 + meta.num_attr_groups * 2)));
							if (reg.size() == 0) {
								fm->reg0 = 0.0;
								fm->regw = 0.0;
								fm->regv = 0.0;
								((fm_learn_mcmc*)fml)->w_lambda.init(fm->regw);
								((fm_learn_mcmc*)fml)->v_lambda.init(fm->regv);
							}
							else if (reg.size() == 1) {
								fm->reg0 = reg[0];
								fm->regw = reg[0];
								fm->regv = reg[0];
								((fm_learn_mcmc*)fml)->w_lambda.init(fm->regw);
								((fm_learn_mcmc*)fml)->v_lambda.init(fm->regv);
							}
							else if (reg.size() == 3) {
								fm->reg0 = reg[0];
								fm->regw = reg[1];
								fm->regv = reg[2];
								((fm_learn_mcmc*)fml)->w_lambda.init(fm->regw);
								((fm_learn_mcmc*)fml)->v_lambda.init(fm->regv);
							}
						}
				}
				else {
					// set the regularization; for standard SGD, groups are not supported
						{
							std::vector<double> reg = cmdline.getDblValues(param_regular);
							assert((reg.size() == 0) || (reg.size() == 1) || (reg.size() == 3));
							if (reg.size() == 0) {
								fm->reg0 = 0.0;
								fm->regw = 0.0;
								fm->regv = 0.0;
							}
							else if (reg.size() == 1) {
								fm->reg0 = reg[0];
								fm->regw = reg[0];
								fm->regv = reg[0];
							}
							else {
								fm->reg0 = reg[0];
								fm->regw = reg[1];
								fm->regv = reg[2];
							}
						}
				}
		
			fm_learn_sgd* fmlsgd = dynamic_cast<fm_learn_sgd*>(fml);
			if (fmlsgd) {
				// set the learning rates (individual per layer)
					{
						std::vector<double> lr = cmdline.getDblValues(param_learn_rate);
						assert((lr.size() == 1) || (lr.size() == 3));
						if (lr.size() == 1) {
							fmlsgd->learn_rate = lr[0];
							fmlsgd->learn_rates.init(lr[0]);
						}
						else {
							fmlsgd->learn_rate = 0;
							fmlsgd->learn_rates(0) = lr[0];
							fmlsgd->learn_rates(1) = lr[1];
							fmlsgd->learn_rates(2) = lr[2];
						}
					}
			}
		



				// () learn		
				fml->learn(train, test);

				// () Prediction at the end  (not for mcmc and als)
				if (cmdline.getValue(param_method).compare("mcmc")) {
					std::cout << "Final\t" << "Train=" << fml->evaluate(train) << "\tTest=" << fml->evaluate(test) << std::endl;
				}


			}
			catch (std::string &e) {
				std::cerr << std::endl << "ERROR: " << e << std::endl;
			}
			catch (char const* &e) {
				std::cerr << std::endl << "ERROR: " << e << std::endl;
			}
		}
	};
}
