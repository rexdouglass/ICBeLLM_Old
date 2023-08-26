
#Every file with an object will get a text file , this code then sorts those into people and non people. Everything left in that folder has no detection and will need to get looked at again

#https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
library(tidyverse)
files <- list.files(path = "/home/skynet3/runs/detect/", full.names = T, recursive = TRUE)
length(files)

object_detections <- list()
library(data.table)
i=1
for(file in files){
  i+i+1
  x <- fread(file, sep=" ", header=F) #read_delim(file, delim=" ", col_names=F)
  x$file_path <- file
  x$file <- basename(x$file_path)
  x$file_noextension <-  tools::file_path_sans_ext(x$file)
  object_detections[[file]] <- x
  if((i %% 100) == 0 ){print(i)}
}

df <- rbindlist(object_detections, fill=TRUE) %>% distinct()
dim(df)
length(unique(df$file_path))

table(df$V1)

files_people <- df %>% filter(V1==0) %>% dplyr::pull(file_noextension) %>% unique()
files_nopeople <- setdiff(unique(df$file_noextension), files_people) %>% unique()

files_to_move <- list.files(path = "/mnt/8tb_b/rex_recovered_from_laptop_drives_sorted/", full.names = T, recursive = TRUE)
files_to_move_df <- data.frame(file_path=files_to_move)
files_to_move_df$file <- basename(files_to_move_df$file_path)
files_to_move_df$file_noextension <-  tools::file_path_sans_ext(files_to_move_df$file)

files_to_move_df$people <- files_to_move_df$file_noextension %in% files_people
files_to_move_df$files_nopeople <- files_to_move_df$file_noextension %in% files_nopeople

files_to_move_df_people <- files_to_move_df %>% filter(people) 
for(i in 1:nrow(files_to_move_df_people)){
  try({
    file.rename(from = files_to_move_df_people$file_path[i],  to = paste0("/mnt/8tb_b/rex_recovered_from_laptop_drives_finalized/people/", files_to_move_df_people$file[i] ) )
  })
} #This gets slower and slower each time because it's erroring on each of those files that have already been processed

files_to_move_df_nopeople <- files_to_move_df %>% filter(files_nopeople) 
for(i in 1:nrow(files_to_move_df_nopeople)){
  try({
    file.rename(from = files_to_move_df_nopeople$file_path[i],  to = paste0("/mnt/8tb_b/rex_recovered_from_laptop_drives_finalized/no_people/", files_to_move_df_nopeople$file[i] ) )
  })
}
