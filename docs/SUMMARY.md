```dataviewjs
const folder = "recieved_notes/Glioma and Sleep";
const pages = dv.pages(`"${folder}"`);

for (let page of pages) {
  const filePath = page.file.path;
  const title = page.title ?? page.file.name;

  // 讀取檔案內容
  const content = await app.vault.read(app.vault.getAbstractFileByPath(filePath));
  
  // 抓取 # 中文摘要 區塊
  const match = content.match(/# 中文摘要.*?\n([\s\S]+?)(?=\n#|$)/);
  const summary = match ? match[1].trim() : "（無摘要）";

  // 顯示為可拖曳的 wikilink
  dv.paragraph(`### [[${page.file.name}|${title}]]`);
  dv.paragraph(summary);
  dv.paragraph("---");
}
