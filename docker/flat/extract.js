"use strict";

/* jshint esversion: 6 */
/* jshint node: true */

/********************************************************************************
 * This utility is used for extracting data from output of experiments.
 */


const fs = require("fs");

const sizes = ["single", "simple", "complex", "huge"];
const types = ["type-b", "type-e", "type-r", "type-be", "type-br", "type-er", "type-ber"];
const algos = ["loda", "lof", "ifor"];

const res = [];
for (const type of types) {
    for (const size of sizes) {
        let s = `${type} & ${size} `;
        for (const algo of algos) {
            const fname = `./csv_normalized_hours.${size}.${type}.${algo}.txt`;
            const content = fs.readFileSync(fname, { encoding: "utf8" })
                .split("\n")
                .map(x => x.trim())
                .filter(x => x.length > 0)
                .pop();
            // console.log(fname, content)
            try {
                const f1 = JSON.parse(content).f1;
                res.push({ size, type, algo, f1 });
                s += `& ` + f1.toFixed(4);
            } catch (e) {
                s += `& ERROR `;
            }
        }
        s += " \\\\";
        console.log(s);
    }
}

// for (const r of res) {
//     console.log()
// }
