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

#include <msclr\marshal_cppstd.h>

using namespace System;
using namespace System::Collections::Generic;
using namespace msclr::interop;
using namespace System::Runtime::InteropServices;

namespace LibFm {

	public ref class FeatureVector
	{
	public:
		float target;
		List<int> ^featureIds;
		List<float> ^featureValues;
		
		String^ ToStringLine()
		{
			String^ line = target.ToString();

			for (int i = 0; i < featureIds->Count; i++)
			{
				line += " " + featureIds[i] + ":" + featureValues[i];
			}

			return line;
		};
	};

	public ref class LibFmManager
	{
	private:
		FmData *train;
		FmData *test;
		fm_model *fm;
		fm_learn* fml;
		CMDLine *cmdline;

	public:
		void CreateTrainSet(List<String^> ^featVectors, float minTarget, float maxTarget, UInt64 numFeatValues, int numFeatures)
		{
			DATA_FLOAT min = static_cast<DATA_FLOAT>(minTarget);
			DATA_FLOAT max = static_cast<DATA_FLOAT>(maxTarget);

			train->initialize(min, max, featVectors->Count, numFeatValues, numFeatures);
			
			for (int i = 0; i < featVectors->Count; i++)
			{
				std::string stdFeatVetor = marshal_as<std::string>(featVectors[i]);
				train->addFeatureVecor(stdFeatVetor);
			}
			
			train->finalizeData();

			test->initialize(min, max, 1, 2, 2);
			std::string sampleTest = "5 0:1 1:1";
			std::string sampleTest2 = "2 2:1 3:1";
			std::string sampleTest3 = "1 1:1 4:1";
			std::string sampleTest4 = "5 5:1 6:1";
			std::string sampleTest5 = "5 7:1 8:1";

			test->addFeatureVecor(sampleTest);
			//test->addFeatureVecor(sampleTest2);
			//test->addFeatureVecor(sampleTest3);
			//test->addFeatureVecor(sampleTest4);
			//test->addFeatureVecor(sampleTest5);

			test->finalizeData();

			SetupModel();

		};

		void Train()
		{
			fml->learn(*train, *test);
		}

		void Clear()
		{
			delete train;
			delete test;
			delete fm;
			delete fml;
		}

		double Predict(FeatureVector ^fv)
		{
			sparse_row<FM_FLOAT> sample;
			sparse_entry<FM_FLOAT> *features = new sparse_entry<FM_FLOAT>[fv->featureIds->Count];
			for (int i = 0; i < fv->featureIds->Count; i++)
			{
				features[i].id = fv->featureIds[i];		// here int is assigned to uint
				features[i].value = fv->featureValues[i]; // here FM_FLOAT is casted to float
			}
			sample.data = features;
			sample.size = fv->featureIds->Count;

			return fm->predict(sample);
		};

		void SetupModel()
		{
			const std::string param_init_stdev = "init_stdev";
			const std::string param_dim = "dim";
			const std::string param_method = "method";
			const std::string param_num_iter = "iter";
			const std::string param_do_sampling = "do_sampling";
			const std::string param_do_multilevel = "do_multilevel";
			const std::string param_learn_rate = "learn_rate";
			const std::string param_regular = "regular";

			uint num_all_attribute = train->num_feature;

			DataMetaInfo meta_main(num_all_attribute);
			//if (cmdline->hasParameter(param_meta_file)) {
			//	meta_main.loadGroupsFromFile(cmdline->getValue(param_meta_file));
			//}

			// build the joined meta table
			for (uint r = 0; r < train->relation.dim; r++) {
				train->relation(r).data->attr_offset = num_all_attribute;
				num_all_attribute += train->relation(r).data->num_feature;
			}
			DataMetaInfo meta(num_all_attribute);
			{
				meta.num_attr_groups = meta_main.num_attr_groups;
				//for (uint r = 0; r < relation->dim; r++) {
				//	meta.num_attr_groups += relation(r)->meta->num_attr_groups;
				//}
				meta.num_attr_per_group.setSize(meta.num_attr_groups);
				meta.num_attr_per_group.init(0);
				for (uint i = 0; i < meta_main.attr_group.dim; i++) {
					meta.attr_group(i) = meta_main.attr_group(i);
					meta.num_attr_per_group(meta.attr_group(i))++;
				}

				uint attr_cntr = meta_main.attr_group.dim;
				uint attr_group_cntr = meta_main.num_attr_groups;
				/*
				for (uint r = 0; r < relation->dim; r++) {
					for (uint i = 0; i < relation(r)->meta->attr_group.dim; i++) {
						meta.attr_group(i + attr_cntr) = attr_group_cntr + relation(r)->meta->attr_group(i);
						meta.num_attr_per_group(attr_group_cntr + relation(r)->meta->attr_group(i))++;
					}
					attr_cntr += relation(r)->meta->attr_group.dim;
					attr_group_cntr += relation(r)->meta->num_attr_groups;
				}
				*/

			}
			meta.num_relations = train->relation.dim;

			try 
			{
			
				fm = new fm_model();
				// (2) Setup the factorization machine
				{
					fm->num_attribute = num_all_attribute;
					fm->init_stdev = cmdline->getValue(param_init_stdev, 0.1);
					// set the number of dimensions in the factorization
					{
						std::vector<int> dim = cmdline->getIntValues(param_dim);
						assert(dim.size() == 3);
						fm->k0 = dim[0] != 0;
						fm->k1 = dim[1] != 0;
						fm->num_factor = dim[2];
					}
					fm->init();

				}


				// (3) Setup the learning method:
				if (!cmdline->getValue(param_method).compare("sgd")) {
					fml = new fm_learn_sgd_element();
					((fm_learn_sgd*)fml)->num_iter = cmdline->getValue(param_num_iter, 100);

				}
				else if (!cmdline->getValue(param_method).compare("mcmc")) {
					fm->w.init_normal(fm->init_mean, fm->init_stdev);
					fml = new fm_learn_mcmc_simultaneous();
					//fml->validation = validation;
					((fm_learn_mcmc*)fml)->num_iter = cmdline->getValue(param_num_iter, 100);
					((fm_learn_mcmc*)fml)->num_eval_cases = 0; //cmdline->getValue(param_num_eval_cases, test.num_cases);

					((fm_learn_mcmc*)fml)->do_sample = cmdline->getValue(param_do_sampling, true);
					((fm_learn_mcmc*)fml)->do_multilevel = cmdline->getValue(param_do_multilevel, true);
				}
				else {
					throw "unknown method";
				}
				fml->fm = fm;
				fml->max_target = train->max_target; //TODO
				fml->min_target = train->min_target;	//TODO
				fml->meta = &meta;
				if (!cmdline->getValue("task").compare("r")) {
					fml->task = 0;
				}
				else if (!cmdline->getValue("task").compare("c")) {
					fml->task = 1;
					for (uint i = 0; i < train->target.dim; i++) { if (train->target(i) <= 0.0) { train->target(i) = -1.0; } else { train->target(i) = 1.0; } }
					for (uint i = 0; i < test->target.dim; i++) { if (test->target(i) <= 0.0) { test->target(i) = -1.0; } else { test->target(i) = 1.0; } }
				}
				else {
					throw "unknown task";
				}


				fml->init();
				

				if (!cmdline->getValue(param_method).compare("mcmc")) {
					// set the regularization; for als and mcmc this can be individual per group
						{
							std::vector<double> reg = cmdline->getDblValues(param_regular);
							//assert((reg.size() == 0) || (reg.size() == 1) || (reg.size() == 3) || (reg.size() == (1 + meta.num_attr_groups * 2)));
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
							std::vector<double> reg = cmdline->getDblValues(param_regular);
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
						std::vector<double> lr = cmdline->getDblValues(param_learn_rate);
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
			}
			catch (std::string &e) {
				std::cerr << std::endl << "ERROR: " << e << std::endl;
			}
			catch (char const* &e) {
				std::cerr << std::endl << "ERROR: " << e << std::endl;
			}
		}



		
		void Setup(List<String^> ^params)
		{
			
			std::vector<std::string> paramsArray(params->Count);

			for (int i = 0; i < params->Count; i++)
			{
				paramsArray[i] = msclr::interop::marshal_as<std::string>(params[i]);
			}
			
			try {
				cmdline = new CMDLine(params->Count, paramsArray);
				const std::string param_task = cmdline->registerParameter("task", "r=regression, c=binary classification [MANDATORY]");
				const std::string param_train_file = cmdline->registerParameter("train", "filename for training data [MANDATORY]");
				const std::string param_test_file = cmdline->registerParameter("test", "filename for test data [MANDATORY]");

				const std::string param_dim = cmdline->registerParameter("dim", "'k0,k1,k2': k0=use bias, k1=use 1-way interactions, k2=dim of 2-way interactions; default=1,1,8");
				const std::string param_regular = cmdline->registerParameter("regular", "'r0,r1,r2' for SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way regularization");
				const std::string param_init_stdev = cmdline->registerParameter("init_stdev", "stdev for initialization of 2-way factors; default=0.1");
				const std::string param_num_iter = cmdline->registerParameter("iter", "number of iterations; default=100");
				const std::string param_learn_rate = cmdline->registerParameter("learn_rate", "learn_rate for SGD; default=0.1");

				const std::string param_method = cmdline->registerParameter("method", "learning method (SGD, SGDA, ALS, MCMC); default=MCMC");

				const std::string param_num_eval_cases = "num_eval_cases";
				const std::string param_do_sampling = "do_sampling";
				const std::string param_do_multilevel = "do_multilevel";

				// Seed
				//long int seed = cmdline->getValue(param_seed, time(NULL));
				srand(time(NULL));

				if (!cmdline->hasParameter(param_method)) { 
					cmdline->setValue(param_method, "mcmc"); 
				}
				if (!cmdline->hasParameter(param_init_stdev)) { cmdline->setValue(param_init_stdev, "0.1"); }
				if (!cmdline->hasParameter(param_dim)) { 
					cmdline->setValue(param_dim, "1,1,8"); 
				}

				if (!cmdline->getValue(param_method).compare("als")) { // als is an mcmc without sampling and hyperparameter inference
					cmdline->setValue(param_method, "mcmc");
					if (!cmdline->hasParameter(param_do_sampling)) { cmdline->setValue(param_do_sampling, "0"); }
					if (!cmdline->hasParameter(param_do_multilevel)) { cmdline->setValue(param_do_multilevel, "0"); }
				}

				// (1) Load the data
				std::cout << "Loading train...\t" << std::endl;
				train = new FmData(0,
					!(!cmdline->getValue(param_method).compare("mcmc")), // no original data for mcmc
					!(!cmdline->getValue(param_method).compare("sgd") || !cmdline->getValue(param_method).compare("sgda")) // no transpose data for sgd, sgda
					);
				//train.load(cmdline->getValue(param_train_file));

				std::cout << "Loading test... \t" << std::endl;
				test = new FmData(0,
					!(!cmdline->getValue(param_method).compare("mcmc")), // no original data for mcmc
					!(!cmdline->getValue(param_method).compare("sgd") || !cmdline->getValue(param_method).compare("sgda")) // no transpose data for sgd, sgda
					);
				//test.load(cmdline->getValue(param_test_file));

				
		


			
				// () learn		
				//fml->learn(*train, test);

				// () Prediction at the end  (not for mcmc and als)
				//if (cmdline->getValue(param_method).compare("mcmc")) {
				//	std::cout << "Final\t" << "Train=" << fml->evaluate(*train) << "\tTest=" << fml->evaluate(test) << std::endl;
				//}


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
