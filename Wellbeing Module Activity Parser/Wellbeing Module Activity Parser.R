activity_file <- read.csv('Data/wbact_1013.dat')
activity_file_non_wb <- read.csv('Data/atusact_0319.dat')

activity_file_modified <- merge(activity_file, activity_file_non_wb, by = c("TUCASEID", "TUACTIVITY_N"))

orig_cols <- colnames(activity_file_modified)[c(2:ncol(activity_file_modified))]


activity_file_modified <- data.frame(do.call(plyr::rbind.fill.matrix,
               by(activity_file_modified, activity_file_modified$TUCASEID, function(x) t(c(x[1, "TUCASEID"], as.vector(t(x[names(x) != "TUCASEID"])))))
))

colnames(activity_file_modified) <- c("TUCASEID", paste0(rep(orig_cols, 3), "_", gl(3, length(orig_cols))))

respondent_file <- subset(read.csv('Data/wbresp_1013.dat'), select = -TULINENO)

final <- merge(respondent_file, activity_file_modified, by = "TUCASEID")
final_grouped_health <- final
final_grouped_health$WEGENHTH[final_grouped_health$WEGENHTH > 3] <- "Bad"
final_grouped_health$WEGENHTH[final_grouped_health$WEGENHTH <= 3] <- "Good"
write.csv(final, "Data/grouped.csv", row.names = F)
write.csv(final_grouped_health, "Data/grouped_grouped_health.csv", row.names = F)
