'''
several running examples, run with
python3 runGan.py 1 # the last number is the run case number # 最後の数字は実行ケース番号です

runcase == 1    inference a trained model   トレーニングされたモデルを推論する
runcase == 2    calculate the metrics, and save the numbers in csv  メトリクスを計算し、数値を CSV に保存します
runcase == 3    training TecoGAN   TecoGANのトレーニング
runcase == 4    training FRVSR     トレーニング FRVSR
runcase == ...  coming... data preparation and so on...   これから…データの準備など…
'''
import os, subprocess, sys, datetime, signal, shutil

runcase = int(sys.argv[1])
print ("Testing test case %d" % runcase)

def preexec(): # Don't forward signals. # シグナルを転送しないでください。
    os.setpgrp()
    
def mycall(cmd, block=False):
    if not block:
        return subprocess.Popen(cmd)
    else:
        return subprocess.Popen(cmd, preexec_fn = preexec)
    
def folder_check(path):
    try_num = 1
    oripath = path[:-1] if path.endswith('/') else path
    while os.path.exists(path):
        print("Delete existing folder " + path + "?(Y/N)")
        decision = input()
        if decision == "Y":
            shutil.rmtree(path, ignore_errors=True)
            break
        else:
            path = oripath + "_%d/"%try_num
            try_num += 1
            print(path)
    
    return path

if( runcase == 0 ): # download inference data, trained models # 推論データ、トレーニング済みモデルをダウンロードする
    # download the trained model # トレーニング済みモデルをダウンロードする
    if(not os.path.exists("./model/")): os.mkdir("./model/")
    cmd1 = "wget https://ge.in.tum.de/download/data/TecoGAN/model.zip -O model/model.zip;"
    cmd1 += "unzip model/model.zip -d model; rm model/model.zip"
    subprocess.call(cmd1, shell=True)
    
    # download some test data # テストデータをダウンロードする
    cmd2 = "wget https://ge.in.tum.de/download/data/TecoGAN/vid3_LR.zip -O LR/vid3.zip;"
    cmd2 += "unzip LR/vid3.zip -d LR; rm LR/vid3.zip"
    subprocess.call(cmd2, shell=True)
    
    cmd2 = "wget https://ge.in.tum.de/download/data/TecoGAN/tos_LR.zip -O LR/tos.zip;"
    cmd2 += "unzip LR/tos.zip -d LR; rm LR/tos.zip"
    subprocess.call(cmd2, shell=True)
    
    # download the ground-truth data # グラウンドトゥルースデータをダウンロードする
    if(not os.path.exists("./HR/")): os.mkdir("./HR/")
    cmd3 = "wget https://ge.in.tum.de/download/data/TecoGAN/vid4_HR.zip -O HR/vid4.zip;"
    cmd3 += "unzip HR/vid4.zip -d HR; rm HR/vid4.zip"
    subprocess.call(cmd3, shell=True)
    
    cmd3 = "wget https://ge.in.tum.de/download/data/TecoGAN/tos_HR.zip -O HR/tos.zip;"
    cmd3 += "unzip HR/tos.zip -d HR; rm HR/tos.zip"
    subprocess.call(cmd3, shell=True)
    
elif( runcase == 1 ): # inference a trained model # トレーニング済みモデルを推論する
    
    dirstr = './results/' # the place to save the results # 結果を保存する場所
    testpre = ['calendar'] # the test cases # テストケース

    if (not os.path.exists(dirstr)): os.mkdir(dirstr)
    
    # run these test cases one by one: # これらのテスト ケースを 1 つずつ実行します。
    for nn in range(len(testpre)):
        cmd1 = ["python3", "main.py",
            "--cudaID", "0",            # set the cudaID here to use only one GPU # GPU を 1 つだけ使用するようにここで cudaID を設定します
            "--output_dir",  dirstr,    # Set the place to put the results. # 結果を配置する場所を設定します。
            "--summary_dir", os.path.join(dirstr, 'log/'), # Set the place to put the log.  # ログを置く場所を設定します。
            "--mode","inference", 
            "--input_dir_LR", os.path.join("./LR/", testpre[nn]),   # the LR directory
            #"--input_dir_HR", os.path.join("./HR/", testpre[nn]),  # the HR directory
            # one of (input_dir_HR,input_dir_LR) should be given
            # (input_dir_HR,input_dir_LR) のいずれかを指定する必要があります
            "--output_pre", testpre[nn], # the subfolder to save current scene, optional # 現在のシーンを保存するサブフォルダー (オプション)
            "--num_resblock", "16",  # our model has 16 residual blocks,  # 私たちのモデルには 16 個の残差ブロックがあります。
            # the pre-trained FRVSR and TecoGAN mini have 10 residual blocks
            # 事前トレーニングされた FRVSR と TecoGAN mini には 10 個の残差ブロックがあります
            "--checkpoint", './model/TecoGAN',  # the path of the trained model, # トレーニングされたモデルのパス、
            "--output_ext", "png"               # png is more accurate, jpg is smaller # png の方が正確ですが、jpg の方が小さいです
        ]
        mycall(cmd1).communicate()

elif( runcase == 2 ): # calculate all metrics, and save the csv files, should use png# すべてのメトリクスを計算し、CSV ファイルを保存します。png を使用する必要があります

    testpre = ["calendar"] # just put more scenes to evaluate all of them # すべてのシーンを評価するには、さらにシーンを追加するだけです
    dirstr = './results/'  # the outputs
    tarstr = './HR/'       # the GT

    tar_list = [(tarstr+_) for _ in testpre]
    out_list = [(dirstr+_) for _ in testpre]
    cmd1 = ["python3", "metrics.py",
        "--output", dirstr+"metric_log/",
        "--results", ",".join(out_list),
        "--targets", ",".join(tar_list),
    ]
    mycall(cmd1).communicate()
    
elif( runcase == 3 ): # Train TecoGAN
    '''
    In order to use the VGG as a perceptual loss,
    we download from TensorFlow-Slim image classification model library:
    https://github.com/tensorflow/models/tree/master/research/slim    
    VGGを知覚損失として使用するには、
    TensorFlow-Slim 画像分類モデル ライブラリからダウンロードします。
    https://github.com/tensorflow/models/tree/master/research/slim
    '''
    VGGPath = "model/" # the path for the VGG model, there should be a vgg_19.ckpt inside# VGG モデルのパス。中には vgg_19.ckpt があるはずです
    VGGModelPath = os.path.join(VGGPath, "vgg_19.ckpt")
    if(not os.path.exists(VGGPath)): os.mkdir(VGGPath)
    if(not os.path.exists(VGGModelPath)):
        # Download the VGG 19 model from  # VGG 19 モデルを次からダウンロードします。
        print("VGG model not found, downloading to %s"%VGGPath)
        cmd0 = "wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz -O " + os.path.join(VGGPath, "vgg19.tar.gz")
        cmd0 += ";tar -xvf " + os.path.join(VGGPath,"vgg19.tar.gz") + " -C " + VGGPath + "; rm "+ os.path.join(VGGPath, "vgg19.tar.gz")
        subprocess.call(cmd0, shell=True)
        
    '''
    Use our pre-trained FRVSR model. If you want to train one, try runcase 4, and update this path by:
    FRVSRModel = "ex_FRVSRmm-dd-hh/model-500000"
    事前トレーニングされた FRVSR モデルを使用します。トレーニングしたい場合は、ランケース 4 を試し、次のようにこのパスを更新します。
    FRVSRModel = "ex_FRVSRmm-dd-hh/model-500000"
    '''
    FRVSRModel = "model/ourFRVSR" 
    if(not os.path.exists(FRVSRModel+".data-00000-of-00001")):
        # Download our pre-trained FRVSR model # 事前トレーニングされた FRVSR モデルをダウンロードする
        print("pre-trained FRVSR model not found, downloading")
        cmd0 = "wget http://ge.in.tum.de/download/2019-TecoGAN/FRVSR_Ours.zip -O model/ofrvsr.zip;"
        cmd0 += "unzip model/ofrvsr.zip -d model; rm model/ofrvsr.zip"
        subprocess.call(cmd0, shell=True)
    
    TrainingDataPath = "/mnt/netdisk/video_data/" 
    
    '''Prepare Training Folder'''
    # path appendix, manually define it, or use the current datetime, now_str = "mm-dd-hh"
    # パスの付録。手動で定義するか、現在の日時を使用します。now_str = "mm-dd-hh"
    now_str = datetime.datetime.now().strftime("%m-%d-%H")
    train_dir = folder_check("ex_TecoGAN%s/"%now_str)
    # train TecoGAN, loss = l2 + VGG54 loss + A spatio-temporal Discriminator
    # TecoGAN を訓練する、損失 = l2 + VGG54 損失 + 時空間弁別器
    cmd1 = ["python3", "main.py",
        "--cudaID", "0", # set the cudaID here to use only one GPU # GPU を 1 つだけ使用するようにここで cudaID を設定します
        "--output_dir", train_dir, # Set the place to save the models. # モデルを保存する場所を設定します。
        "--summary_dir", os.path.join(train_dir,"log/"), # Set the place to save the log.  # ログの保存場所を設定します。
        "--mode","train",
        "--batch_size", "4" , # small, because GPU memory is not big # GPUメモリが大きくないので小さい
        "--RNN_N", "10" , # train with a sequence of RNN_N frames, >6 is better, >10 is not necessary # RNN_N フレームのシーケンスを使用してトレーニングします。6 を超える方が優れており、10 を超える必要はありません
        "--movingFirstFrame", # a data augmentation # データの拡張
        "--random_crop",
        "--crop_size", "32",
        "--learning_rate", "0.00005",
        # -- learning_rate step decay, here it is not used --
        # -- learning_rate ステップ減衰、ここでは使用されません --
        "--decay_step", "500000", 
        "--decay_rate", "1.0", # 1.0 means no decay # 1.0 は減衰がないことを意味します
        "--stair",
        "--beta", "0.9", # ADAM training parameter beta # ADAMトレーニングパラメータベータ版
        "--max_iter", "500000", # 500k or more, the one we present is trained for 900k # 500k 以上、私たちが提示するものは 900k でトレーニングされています
        "--save_freq", "10000", # the frequency we save models # モデルを保存する頻度
        # -- network architecture parameters --
        "--num_resblock", "16", # FRVSR and TecoGANmini has num_resblock as 10. The TecoGAN has 16. # FRVSR と TecoGANmini の num_resblock は 10 です。TecoGAN の num_resblock は 16 です
        # -- VGG loss, disable with vgg_scaling < 0
        # -- VGG 損失、vgg_scaling < 0 で無効化
        "--vgg_scaling", "0.2",
        "--vgg_ckpt", VGGModelPath, # necessary if vgg_scaling > 0 # vgg_scaling > 0 の場合に必要
    ]
    '''Video Training data:
    please udate the TrainingDataPath according to ReadMe.md
    input_video_pre is hard coded as scene in dataPrepare.py at line 142
    str_dir is the starting index for training data
    end_dir is the ending index for training data
    end_dir+1 is the starting index for validation data
    end_dir_val is the ending index for validation data
    max_frm should be duration (in dataPrepare.py) -1
    queue_thread: how many cpu can be used for loading data when training
    name_video_queue_capacity, video_queue_capacity: how much memory can be used
    ビデオトレーニングデータ:
    ReadMe.md に従って TrainingDataPath を更新してください。
    input_video_pre は、dataPrepare.py の 142 行目でシーンとしてハードコードされています。
    str_dir はトレーニング データの開始インデックスです。
    end_dir はトレーニング データの終了インデックスです。
    end_dir+1 は検証データの開始インデックスです。
    end_dir_val は検証データの終了インデックスです。
    max_frm は期間 (dataPrepare.py 内) -1 である必要があります
    queue_thread: トレーニング時にデータのロードに使用できる CPU の数
    name_video_queue_capacity、video_queue_capacity: 使用できるメモリの量
    '''
    cmd1 += [
        "--input_video_dir", TrainingDataPath, 
        "--input_video_pre", "scene",
        "--str_dir", "2000",
        "--end_dir", "2250",
        "--end_dir_val", "2290",
        "--max_frm", "119",
        # -- cpu memory for data loading --
        # -- データ読み込み用の CPU メモリ --
        "--queue_thread", "12",# Cpu threads for the data. >4 to speedup the training # データの CPU スレッド。トレーニングをスピードアップするには >4
        "--name_video_queue_capacity", "1024",
        "--video_queue_capacity", "1024",
    ]
    '''
    loading the pre-trained model from FRVSR can make the training faster
    --checkpoint, path of the model, here our pre-trained FRVSR is given
    --pre_trained_model,  to continue an old (maybe accidentally stopeed) training, 
        pre_trained_model should be false, and checkpoint should be the last model such as 
        ex_TecoGANmm-dd-hh/model-xxxxxxx
        To start a new and different training, pre_trained_model is True. 
        The difference here is 
        whether to load the whole graph icluding ADAM training averages/momentums/ and so on
        or just load existing pre-trained weights.
        FRVSR から事前トレーニングされたモデルをロードすると、トレーニングが高速化されます。
    --checkpoint、モデルのパス、ここでは事前トレーニングされた FRVSR が与えられます
    --pre_trained_model、古い (おそらく誤って停止した) トレーニングを続行するには、
    pre_trained_model は false である必要があり、チェックポイントは次のような最後のモデルである必要があります。
    ex_TecoGANmm-dd-hh/モデル-xxxxxxx
    新しく異なるトレーニングを開始するには、pre_trained_model を True にします。
    ここでの違いは、
    ADAM トレーニングの平均/運動量などを含むグラフ全体をロードするかどうか
    または、既存の事前トレーニングされた重みをロードするだけです。
    '''
    cmd1 += [ # based on a pre-trained FRVSR model. Here we want to train a new adversarial training
        "--pre_trained_model", # True
        "--checkpoint", FRVSRModel,
    ]
    
    # the following can be used to train TecoGAN continuously
    # 以下を使用して TecoGAN を継続的にトレーニングできます
    # old_model = "model/ex_TecoGANmm-dd-hh/model-xxxxxxx" 
    # old_model = "モデル/ex_TecoGANmm-dd-hh/モデル-xxxxxxx"
    # cmd1 += [ # Here we want to train continuously # ここで継続的にトレーニングしたい
    #     "--nopre_trained_model", # False
    #     "--checkpoint", old_model,
    # ]
    
    ''' parameters for GAN training '''
    cmd1 += [
        "--ratio", "0.01",  # the ratio for the adversarial loss from the Discriminator to the Generator 
                            # Discriminator から Generator までの敵対的損失の比率
        "--Dt_mergeDs",     # if Dt_mergeDs == False, only use temporal inputs, so we have a temporal Discriminator
                            # Dt_mergeDs == False の場合、時間入力のみを使用するため、時間弁別器が存在します。
                            # else, use both temporal and spatial inputs, then we have a Dst, the spatial and temporal Discriminator
        # それ以外の場合は、時間入力と空間入力の両方を使用すると、Dst (空間的および時間的識別子) が得られます。
    ]
    ''' if the generator is pre-trained, to fade in the discriminator is usually more stable.
    the weight of the adversarial loss will be weighed with a weight, started from Dt_ratio_0, 
    and increases until Dt_ratio_max, the increased value is Dt_ratio_add per training step
    For example, fading Dst in smoothly in the first 4k steps is 
    "--Dt_ratio_max", "1.0", "--Dt_ratio_0", "0.0", "--Dt_ratio_add", "0.00025"
    ジェネレーターが事前にトレーニングされている場合、通常はディスクリミネーターをフェードインする方が安定します。
    敵対的損失の重みは、Dt_ratio_0 から始まる重みで重み付けされます。
    Dt_ratio_max まで増加します。増加した値はトレーニング ステップごとの Dt_ratio_add です。
    たとえば、最初の 4k ステップで Dst をスムーズにフェードインするには、次のようにします。
    "--Dt_ratio_max"、"1.0"、"--Dt_ratio_0"、"0.0"、"--Dt_ratio_add"、"0.00025"
    '''
    cmd1 += [ # here, the fading in is disabled  # ここではフェードインが無効になっています
        "--Dt_ratio_max", "1.0",
        "--Dt_ratio_0", "1.0", 
        "--Dt_ratio_add", "0.0", 
    ]
    ''' Other Losses '''
    cmd1 += [
        "--pingpang",           # our Ping-Pang loss
        "--pp_scaling", "0.5",  # the weight of the our bi-directional loss, 0.0~0.5 # 双方向損失の重み、0.0~0.5
        "--D_LAYERLOSS",        # use feature layer losses from the discriminator # 弁別器からの特徴層の損失を使用する
    ]
    
    pid = mycall(cmd1, block=True) 
    try: # catch interruption for training # トレーニングの中断をキャッチ
        pid.communicate()
    except KeyboardInterrupt: # Ctrl + C to stop current training try to save the last model  # Ctrl + C を押して現在のトレーニングを停止し、最後のモデルを保存してみてください
        print("runGAN.py: sending SIGINT signal to the sub process...")
        pid.send_signal(signal.SIGINT)
        # try to save the last model  # 最後のモデルを保存してみる
        pid.communicate()
        print("runGAN.py: finished...")
        
        
elif( runcase == 4 ): # Train FRVSR, loss = l2 warp + l2 content # FRVSR を訓練する、損失 = l2 ワープ + l2 コンテンツ
    now_str = datetime.datetime.now().strftime("%m-%d-%H")
    train_dir = folder_check("ex_FRVSR%s/"%now_str)
    cmd1 = ["python3", "main.py",
        "--cudaID", "0", # set the cudaID here to use only one GPU # GPU を 1 つだけ使用するようにここで cudaID を設定します
        "--output_dir", train_dir, # Set the place to save the models. # モデルを保存する場所を設定します。
        "--summary_dir", os.path.join(train_dir,"log/"), # Set the place to save the log.  # ログの保存場所を設定します。
        "--mode","train",
        "--batch_size", "4" , # small, because GPU memory is not big # GPUメモリが大きくないので小さい
        "--RNN_N", "10" , # train with a sequence of RNN_N frames, >6 is better, >10 is not necessary # RNN_N フレームのシーケンスを使用してトレーニングします。6 を超える方が優れており、10 を超える必要はありません
        "--movingFirstFrame", # a data augmentation # データの拡張
        "--random_crop",
        "--crop_size", "32",
        "--learning_rate", "0.00005",
        # -- learning_rate step decay, here it is not used --
        # -- learning_rate ステップ減衰、ここでは使用されません --
        "--decay_step", "500000", 
        "--decay_rate", "1.0", # 1.0 means no decay # 1.0 は減衰がないことを意味します
        "--stair",
        "--beta", "0.9", # ADAM training parameter beta # ADAMトレーニングパラメータベータ版
        "--max_iter", "500000", # 500k is usually fine for FRVSR, GAN versions need more to be stable # FRVSR には通常 500k で十分ですが、GAN バージョンを安定させるにはさらに多くの値が必要です
        "--save_freq", "10000", # the frequency we save models # モデルを保存する頻度
        # -- network architecture parameters --
        "--num_resblock", "10", # a smaller model # より小さいモデル
        "--ratio", "-0.01",  # the ratio for the adversarial loss, negative means disabled # 敵対的損失の比率、負は無効を意味します
        "--nopingpang",
    ]
    '''Video Training data... Same as runcase 3...'''
    TrainingDataPath = "/mnt/netdisk/video_data/"
    cmd1 += [
        "--input_video_dir", TrainingDataPath, 
        "--input_video_pre", "scene",
        "--str_dir", "2000",
        "--end_dir", "2250",
        "--end_dir_val", "2290",
        "--max_frm", "119",
        # -- cpu memory for data loading -- # -- データ読み込み用の CPU メモリ --
        "--queue_thread", "12",# Cpu threads for the data. >4 to speedup the training # データの CPU スレッド。トレーニングをスピードアップするには >4
        "--name_video_queue_capacity", "1024",
        "--video_queue_capacity", "1024",
    ]
    
    pid = mycall(cmd1, block=True)
    try: # catch interruption for training # トレーニングの中断をキャッチ
        pid.communicate()
    except KeyboardInterrupt: # Ctrl + C to stop current training try to save the last model  # Ctrl + C を押して現在のトレーニングを停止し、最後のモデルを保存してみてください
        print("runGAN.py: sending SIGINT signal to the sub process...")
        pid.send_signal(signal.SIGINT)
        # try to save the last model  # 最後のモデルを保存してみる
        pid.communicate()
        print("runGAN.py: finished...")
