// Push 3: exact simplified MuJoCo exterior reference, units: mm.
// Source: shadow_hand_push3.xml
// SHA-256: 7c90d46de779c5a5292c0421d92f069de7fee593c44bb7eee9fe7e5369365697
// Not a working push-button mechanism: no bore, guide, stop, or spring seat.
// In the simulator the body/plunger collision pair is excluded; positive
// press_mm intentionally makes the solid reference shapes interpenetrate.

$fn = 128;
view = "assembly"; // "assembly", "body", "plunger", "exploded"
press_mm = 0;    // 0 .. 12; simulator travel toward -Z
exploded_gap_mm = 12;

body_d_mm = 52;
body_h_mm = 120;
neck_d_mm = 14;
neck_h_mm = 18;
button_d_mm = 28;
button_h_mm = 10;
stroke_mm = 12;

module push3_body() {
    cylinder(d=body_d_mm, h=body_h_mm);
}

module push3_plunger() {
    union() {
        cylinder(d=neck_d_mm, h=neck_h_mm);
        translate([0, 0, neck_h_mm])
            cylinder(d=button_d_mm, h=button_h_mm);
    }
}

module push3_assembly(press=0, gap=0) {
    color([0.90, 0.93, 0.97]) push3_body();
    color([0.20, 0.36, 0.62])
        translate([0, 0, body_h_mm - press + gap]) push3_plunger();
}

assert(press_mm >= 0 && press_mm <= stroke_mm, "press_mm outside MuJoCo range");
if (view == "body") push3_body();
else if (view == "plunger") push3_plunger();
else if (view == "exploded") push3_assembly(0, exploded_gap_mm);
else push3_assembly(press_mm);
