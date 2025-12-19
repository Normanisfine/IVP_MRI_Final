
# loads both T2 and ADC modalities as separate RGB channels.
# Channel mapping: R=T2, G=ADC, B=average(T2,ADC)


class NPZDualModalityDataset(NPZRawDataset):
    """
    NPZ dataset that loads both T2 and ADC modalities as separate channels.
    Channel mapping: R=T2, G=ADC, B=average(T2,ADC)
    """
    def __init__(
        self,
        folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        truncate_video=-1,
    ):
        super().__init__(
            folder=folder,
            file_list_txt=file_list_txt,
            excluded_videos_list_txt=excluded_videos_list_txt,
            sample_rate=sample_rate,
            truncate_video=truncate_video,
        )

    def get_video(self, idx):
        """
        Load both T2 and ADC modalities and combine them into RGB channels.
        """
        video_name = self.video_names[idx]
        npz_path = os.path.join(self.folder, f"{video_name}.npz")

        npz_data = np.load(npz_path)

        # Extract T2 and ADC images
        if 'imgs_t2' in npz_data and 'imgs_adc' in npz_data:
            t2 = npz_data['imgs_t2'] / 255.0  # (D, H, W)
            adc = npz_data['imgs_adc'] / 255.0  # (D, H, W)
        else:
            single = npz_data['imgs'] / 255.0
            t2 = single
            adc = single

        avg = (t2 + adc) / 2.0
        frames = np.stack([t2, adc, avg], axis=1)  # (D, 3, H, W)

        masks = npz_data['gts']

        if self.truncate_video > 0:
            frames = frames[:self.truncate_video]
            masks = masks[:self.truncate_video]

        vos_frames = []
        for i, frame in enumerate(frames[::self.sample_rate]):
            frame_idx = i * self.sample_rate
            vos_frames.append(VOSFrame(frame_idx, image_path=None, data=torch.from_numpy(frame)))

        video = VOSVideo(video_name, idx, vos_frames)

        segment_loader = NPZSegmentLoader(masks[::self.sample_rate])

        return video, segment_loader
