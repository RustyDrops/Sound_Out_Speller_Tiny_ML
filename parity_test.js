const fs = require('fs');
global.MODEL = JSON.parse(fs.readFileSync('demo/model.json', 'utf8'));
const core = require('./demo/core.js');
const attempts = ['sed','wos','teh','becos','jumpt','fone','riting','said','haus','tre','watur','skool','frend','hapy','nite','lite','dere','dere','wat','wen','wich','hur','ovur','funni','litul','becuse','togethr','scool','peple','animul','techer'];
for (const a of attempts) {
  const r = core.suggest(a, []);
  console.log(a + '|' + r.ranked.slice(0,5).map(x=>x.word).join(',') + '|' + r.exact + '|' + r.margin.toFixed(4) + '|' + core.isAccepted(a, r));
}
