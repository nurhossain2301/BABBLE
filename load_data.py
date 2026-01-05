
from dataset_LB import LBAudioDataset, LBAudioDatasetCaption, LBAudioDatasetFrameBased, LBAudioDatasetMiddleFrame, LBAudioDatasetSpeakerEmbeddingFrameBased, MyCollator, MyCollatorSpeakerEmbedding, MyCollatorAudio
# from dataset_for_zero_shot import LBAudioDataset, LBAudioDatasetMiddleFrame
import torch.utils.data as data_utils
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from tqdm import tqdm


class FilteredDataset(data_utils.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = []
        self.txt_file = "VAD_detection.txt"
        self.f = open(self.txt_file, 'w')
        self.model = load_silero_vad().to('cuda')

        


        for i in range(len(dataset)):
            audio = dataset[i][0]
            self.f.write('target: ' + dataset[i][3] +'\n')
            speech_timestamps = get_speech_timestamps(
                    audio.to('cuda'),
                    self.model,
                    return_seconds=True,  # Return speech timestamps in seconds (default is samples)
                    )
            if len(speech_timestamps) > 0:
                self.indices.append(i)
                self.f.write('prediction: ' + 'speech\n')
            else:
                self.f.write('prediction: ' + 'no_speech\n')
                

        # Precompute indices we want to keep
        # self.indices = [
        #     i for i in range(len(dataset))
        #     if dataset[i][3] != '[]'
        # ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]



def load_data_for_qwen(args):
    test_dataset = LBAudioDatasetMiddleFrame(
        json_file = args.test_json, 
        mode='test',
        apply_1_shot=True,
        audio_length=2,
        # inference_only=args.inference_only,
    )

    test_loader = data_utils.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return test_loader

def load_training_data(args, tokenizer=None):
    # if args.mode=="event":
    #     train_dataset = LBAudioDatasetCaption(
    #         json_file=args.train_json,
    #         mode='train',
    #         apply_1_shot=True,
    #         include_voc_count=True,
    #     )
            
    #     val_dataset = LBAudioDatasetCaption(
    #         json_file=args.val_json,
    #         mode='valid',
    #         apply_1_shot=True,
    #         include_voc_count=True,
    #     )
    #     print(len(train_dataset), len(val_dataset))

    #     my_collator = MyCollator(tokenizer)
    if args.mode=="event":
        train_dataset = LBAudioDataset(
            json_file=args.train_json,
            mode='train',
            apply_1_shot=False,
            include_voc_count=True,
        )
            
        val_dataset = LBAudioDataset(
            json_file=args.val_json,
            mode='valid',
            apply_1_shot=True,
            include_voc_count=True,
        )
        # print(len(train_dataset), len(val_dataset))

        my_collator = MyCollator(tokenizer)
    elif args.mode=="frame":
        if args.frame_size==2.0:
            train_dataset = LBAudioDatasetFrameBased(
                json_file=args.train_json,
                mode='train', 
                apply_1_shot=True,
                audio_length=2,
                target_length=2,
            )
    
            val_dataset = LBAudioDatasetFrameBased(
                json_file=args.val_json,
                mode='valid',
                apply_1_shot=True,
                audio_length=2,
                target_length=2,
            )
        else: # frame_size == 0.1
            train_dataset = LBAudioDatasetMiddleFrame(
                tokenizer=tokenizer,
                json_file=args.train_json,
                mode='train', 
                apply_1_shot=False,
                audio_length=30,
            )
            # train_dataset = FilteredDataset(train_dataset)
            
            val_dataset = LBAudioDatasetMiddleFrame(
                tokenizer=tokenizer,
                json_file=args.val_json,
                mode='valid',
                apply_1_shot=False,
                audio_length=30,
            )

            # val_dataset = FilteredDataset(val_dataset)
    
        my_collator = MyCollator(tokenizer)
        
    else: # args.mode=="frame_spk_emb"
        train_dataset = LBAudioDatasetSpeakerEmbeddingFrameBased(
            json_file=args.train_json,
            mode='train', 
            apply_1_shot=True,
            audio_length=2,
            target_length=2,
            )
        
        val_dataset = LBAudioDatasetSpeakerEmbeddingFrameBased(
            json_file=args.val_json,
            mode='valid',
            apply_1_shot=True,
            audio_length=2,
            target_length=2,
            )
        my_collator = MyCollatorSpeakerEmbedding(tokenizer)

    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collator, num_workers=args.num_workers)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collator, num_workers=args.num_workers)
    # train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # val_loader = data_utils.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader

def load_testing_data(args, tokenizer):
    if args.mode=="event":
        test_dataset = LBAudioDataset(
            json_file = args.test_json, 
            mode='test',
            apply_1_shot=True,
            include_voc_count=True,
        )
        my_collator = MyCollator(tokenizer)
    elif args.mode=="frame":
        if args.frame_size==2.0:
            test_dataset = LBAudioDatasetFrameBased(
                json_file = args.test_json, 
                mode='test',
                apply_1_shot=True,
                audio_length=2,
                target_length=2,
            )
        else: #frame size ==0.1
            test_dataset = LBAudioDatasetMiddleFrame(
                tokenizer=tokenizer,
                json_file = args.test_json, 
                mode='test',
                apply_1_shot=True,
                audio_length=30,
                # inference_only=args.inference_only,
            )
            # test_dataset = FilteredDataset(test_dataset)
        my_collator = MyCollator(tokenizer)
    else: # args.mode=="frame_spk_emb"
        test_dataset = LBAudioDatasetSpeakerEmbeddingFrameBased(
            json_file = args.test_json,  
            mode='test',
            apply_1_shot=True,
            audio_length=2,
            target_length=2,
        )
    test_loader = data_utils.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn = my_collator)
    return test_loader
