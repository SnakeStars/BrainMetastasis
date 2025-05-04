module BrainMetastasis
    using DataFrames
    using Dates
    using CSV
    using Statistics
    using Plots, StatsPlots
    using MLJ
    using Logging
    using StatisticalMeasuresBase
    using Shapley, CategoricalDistributions
    import Imbalance

    function dataRename!(df::DataFrame)
        rename!(df, [:Sex,
                     :CountOfFractions,
                     :KarnovskiyIndex,
                     :CountOfGM,
                     :SummVol,
                     :Progression,
                     :RMZ,
                     :NMRL,
                     :Melanoma,
                     :RP,
                     :KRR,
                     :OVGM,
                     :OperationGM,
                     :ChemistryTherapy,
                     :TargetTherapy,
                     :NoMedic,
                     :AgeOnCRT,
                     :ReactionTime,
                     :MetastasisTime])
    end

    function cleanPreparing(balancing="no")
        data = CSV.read("completed.csv", DataFrame)
        dataRename!(data)
        y, X = unpack(data, ==(:Progression), rng=1234)
        y = coerce(y, OrderedFactor)
        if(balancing == "oversampling")
            println("Before oversamppling")
            Imbalance.checkbalance(y)
            oversampler = (@load RandomOversampler pkg=Imbalance verbosity=0)()
            mach = machine(oversampler)
            X, y = MLJ.transform(mach, X, y)
            println("After oversamppling")
        elseif(balancing == "SMOTE")
            println("Before SMOTE")
            Imbalance.checkbalance(y)
            oversampler = (@load SMOTE pkg=Imbalance verbosity=0)()
            mach = machine(oversampler)
            X, y = MLJ.transform(mach, X, y)
            println("After SMOTE")
        elseif(balancing == "no")
            println("No balancing")
        end
        Imbalance.checkbalance(y)
        (Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.8, rng=123, multi=true, stratify=y)
        return Xtrain, Xtest, ytrain, ytest
    end

    function PreparingBestModels(X, y)
        # X = DataFrames.select(data, Not(:"Интракраниальная прогрессия"))  # Все колонки, кроме 'target'
        # Y = DataFrames.select(data,"Интракраниальная прогрессия")              # Целевая переменная

        models = []

        
        model = (@load KNNClassifier verbosity=0)()
        K_range = range(model, :K, lower=5, upper=20)
        self_tuning_knn = TunedModel(
            model=model,
            resampling = StratifiedCV(nfolds=10, rng=1234),
            tuning = Grid(),
            range = K_range,
            measure = FScore(; beta=2),
            operation = predict_mode
        )
        mach = machine(self_tuning_knn, X, y)
        fit!(mach, verbosity=0)
        push!(models, mach)

        tree = (@load DecisionTreeClassifier pkg=DecisionTree verbosity=0)()
        m_depth_range = range(tree, :max_depth, lower=-1, upper=80);
        self_tuning_tree = TunedModel(
            model=tree,
            resampling = StratifiedCV(nfolds=10, rng=1234),
            tuning = Grid(),
            range = m_depth_range,
            measure = FScore(; beta=2),
            operation = predict_mode
        )
        mach = machine(self_tuning_tree, X, y)
        fit!(mach, verbosity=0)
        push!(models, mach)

        Forest = (@load RandomForestClassifier pkg=DecisionTree verbosity=0)()
        m_depth_range = range(Forest, :max_depth, lower=-1, upper=80)
        n_sub_range = range(Forest, :n_subfeatures, lower=-1, upper=18)
        n_tree_range = range(Forest, :n_trees, lower=2, upper=100)
        self_tuning_forest = TunedModel(
            model=Forest,
            resampling = StratifiedCV(nfolds=10, rng=1234),
            tuning = RandomSearch(),
            range = [m_depth_range, n_sub_range, n_tree_range],
            measure = FScore(; beta=2),
            operation = predict_mode
        )
        mach = machine(self_tuning_forest, X, y)
        fit!(mach, verbosity=0)
        push!(models, mach)

        SVC = (@load ProbabilisticSVC pkg=LIBSVM verbosity=0)()
        mach = machine(SVC, X, y)
        fit!(mach, verbosity=0)
        push!(models, mach)

        LogClass = (@load LogisticClassifier pkg=MLJLinearModels verbosity=0)()
        mach = machine(LogClass, X, y)
        fit!(mach, verbosity=0)
        push!(models, mach)
        return models
    end

    function modelsReport(best_models, Xvalid, yvalid)
        model_names = []
        bac = []
        jac = []
        f2 = []
        FN = []
    
        for model in best_models
            try
                push!(model_names, report(model).best_history_entry.model)
            catch
                push!(model_names, model.model)
            end
            y = predict_mode(model, Xvalid)
            push!(bac, balanced_accuracy(y, yvalid))
            push!(f2, FScore(beta = 2)(y, yvalid))
            push!(FN, falsenegative(y, yvalid))
        end
    
        df_valid = DataFrame(Models=model_names, balanced_acc=bac, F2=f2, FN=FN)
        return df_valid
    end

    function main()
        Xtrain, Xtest, ytrain, ytest = cleanPreparing("SMOTE")
        models = PreparingBestModels(Xtrain, ytrain)
        modelsReport(models, Xtest, ytest)
    end
end # module BrainMetastasis
