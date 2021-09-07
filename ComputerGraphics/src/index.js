import * as THREE from 'three'

//Hello World
var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

var renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );
var geometry = new THREE.BoxGeometry( 1, 1, 1 );
var material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
var cube = new THREE.Mesh( geometry, material );
//scene.add( cube );

//Add Lines
var line_material = new THREE.LineBasicMaterial({color: 0xff0000})
var line_geometry = new THREE.BufferGeometry();
var line_vertices= new Float32Array([
    -1.0, -1.0, -1.0,
    1.0, 1.0, 1.0
])
line_geometry.setAttribute('position',new THREE.BufferAttribute(line_vertices,3))
var line_mesh = new THREE.Line(line_geometry,line_material);
//scene.add(line_mesh)

//Add Text
var text_loader=new THREE.FontLoader();
var text_geometry;
var text_mesh;
text_loader.load("helvetiker_regular.typeface.json",
    function(font){
        text_geometry=new THREE.TextGeometry("HelloWorld",{
            font:font,
            size:80,
            height:5,
            curveSegments:12
        })
        text_mesh = new THREE.Mesh(text_geometry,material)
        scene.add(text_mesh)
    }
);


// Rendering
camera.position.z = 0;
function animate() {
	requestAnimationFrame( animate );
    line_mesh.rotation.x += 0.01;
    line_mesh.rotation.y += 0.01;
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;
    camera.position.z += 0.5;
	renderer.render( scene, camera );
}
animate();