import movies_converter_MP4 as MyConverterToMP4
import movies_frames_extractor as MyExtractor
import movie_frames_augmentator as MyAugmentator
import frames_split_into_csv as MyMovieCSVSplitter

if __name__ == '__main__':
    print("1. Start converting the movies to MP4 format")
    MyConverterToMP4.convert_files()

    print("2. Start extracting the frames from the movies")
    MyExtractor.movies_extraction()

    print("3. Start augmenting the frames")
    MyAugmentator.apply_augmentation()

    print("4. Split my movie frames into CSV files")
    MyMovieCSVSplitter.movies_frames_to_csv()

    print("5. Combine all CSV files into one")
    MyMovieCSVSplitter.combine_all_csv()

    print("6. Split data into train and test")
    MyMovieCSVSplitter.split_data_to_train_test()