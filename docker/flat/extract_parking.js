"use strict";

/* jshint esversion: 6 */
/* jshint node: true */

const fs = require("fs");

const sizes = ["single", "simple", "complex", "huge"];
const types = ["type-b", "type-e", "type-r", "type-be", "type-br", "type-er", "type-ber"];
const algos = ["loda", "lof", "ifor"];
const csv_type = "csv";
//const csv_type = "csv_normalized_hours";
const decimals = 3;

const res = [];
for (const type of types) {
    for (const size of sizes) {
        let s = `${type} & ${size}`;
        for (const algo of algos) {
            const fname = `./${csv_type}.${size}.${type}.${algo}.txt`;
            const content = fs.readFileSync(fname, { encoding: "utf8" })
                .split("\n")
                .map(x => x.trim())
                .filter(x => x.length > 0)
                .pop();
            // console.log(fname, content)
            try {
                const f1 = JSON.parse(content).f1;
                const precision = JSON.parse(content).precision;
                const recall = JSON.parse(content).recall;
                res.push({ size, type, algo, f1 });
                s += ` & ` + precision.toFixed(decimals);
                s += ` & ` + recall.toFixed(decimals);
                s += ` & ` + f1.toFixed(decimals);
            } catch (e) {
                s += `& ERROR `;
            }
        }
        s += " \\\\";
        console.log(s);
    }
    console.log("\\midrule");
}

// for (const r of res) {
//     console.log()
// }
