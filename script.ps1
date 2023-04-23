$folderPath = "D:\CUFE\Year 4\Semester 2\GP\Project\photogrammetry\ziad\images\snow-man"
$files = Get-ChildItem -Path $folderPath -Filter *.jpg
$count = 1

foreach ($file in $files) {
    Rename-Item $file.FullName -NewName "$count.jpg"
    $count++
}