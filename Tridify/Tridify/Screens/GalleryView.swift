//
//  GalleryView.swift
//  Tridify
//
//  Created by Maged Alosali on 18/05/2023.
//

import SwiftUI
import SceneKit

struct GalleryView: View {
    
    let path = URL(fileURLWithPath: Bundle.main.path(forResource: "point_cloud", ofType: "ply")!)
    @State private var scene: SCNScene?
    @State private var error: Error?
    var body: some View {
        VStack {
            Text ("path: \(path.absoluteString)")
            if let scene = scene {
                SceneView(scene: scene, options: [.autoenablesDefaultLighting, .allowsCameraControl])
                    .frame(width: UIScreen.main.bounds.width, height: UIScreen.main.bounds.height/2)
            }
            else if let error = error {
                Text("Error")
            }
            else {
                ProgressView()
            }
        }
        .onAppear {
            do {
                let load = try SCNScene(url: path)
                scene = load
            } catch {
                self.error = error
            }
    
        }
    }
}

struct GalleryView_Previews: PreviewProvider {
    static var previews: some View {
        GalleryView()
    }
}
