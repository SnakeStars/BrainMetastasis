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
        Xtrain, Xtest, ytrain, ytest = cleanPreparing("no")
        models = PreparingBestModels(Xtrain, ytrain)
        table = modelsReport(models, Xtest, ytest)
        CSV.write("no-balancing.csv", table)
        # ShapleyResearch(models, Xtest, 1, "KNN-oversampling")
        # ShapleyResearch(models, Xtest, 2, "DecisionTree-oversampling")
        # ShapleyResearch(models, Xtest, 3, "RandomForest-oversampling")
        # ShapleyResearch(models, Xtest, 4, "SVC-oversampling")
        # ShapleyResearch(models, Xtest, 5, "Log-oversampling")
    end

    function ShapleyResearch(models, Xvalid, i, pl_tit)
        ϕ = shapley(Xvalid -> predict(models[i], Xvalid), Shapley.MonteCarlo(CPUThreads(), 1024), Xvalid)
        bar_data = []
        k = [string(i) for i in keys(ϕ)]
        for i in ϕ 
            push!(bar_data, mean(abs.(pdf.(i, 1))))
        end
        n = size(bar_data, 1)
        b = bar(
            bar_data,
            yticks=(1:1:n, k),
            ylims=(0, n+1),
            orientation=:horizontal,
            legend=false,
            xlims=(0, 0.3),
            title="Global feature importance",
            xlabel="Mean(abs(Shapley value))",
        )
        A = [pdf.(i, 1) for i in ϕ]
        v = violin(
            A, 
            xticks=(1:1:n, k),
            xlims=(0, n+1),
            legend=false,
            title="Local explanation summary",
            ylabel="SHAP value", 
            permute=(:y, :x),
        )
        fig = plot(
            b,
            v,
            layout=(1, 2),
            plot_title=pl_tit,
            size=(1200, 900),
            margin=(20, :pt)
            )
        savefig(fig, pl_tit)
    end  # function ShapleyResearch
end # module BrainMetastasis

